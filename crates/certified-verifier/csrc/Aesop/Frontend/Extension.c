// Lean compiler output
// Module: Aesop.Frontend.Extension
// Imports: public import Init public meta import Init public import Aesop.Frontend.Extension.Init public import Lean.Meta.Tactic.Simp.Simproc import Lean.Meta.Tactic.Simp.Attr
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
lean_object* l_Lean_Name_beq___boxed(lean_object*, lean_object*);
lean_object* l_Lean_Name_hash___override___boxed(lean_object*);
lean_object* lp_aesop_Aesop_getDeclaredRuleSets___boxed(lean_object*);
lean_object* l_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* l_Lean_MessageData_ofName(lean_object*);
lean_object* l_Lean_throwError___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_Simp_getSimprocExtension_x3f___boxed(lean_object*, lean_object*);
lean_object* l_Lean_Meta_getSimpExtension_x3f___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_get_size(lean_object*);
uint64_t lean_uint64_shift_right(uint64_t, uint64_t);
uint64_t lean_uint64_xor(uint64_t, uint64_t);
size_t lean_uint64_to_usize(uint64_t);
size_t lean_usize_of_nat(lean_object*);
size_t lean_usize_sub(size_t, size_t);
size_t lean_usize_land(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
uint8_t lean_name_eq(lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_RuleSetNameFilter_matchedRuleSetNames(lean_object*);
lean_object* lp_aesop_Aesop_GlobalRuleSet_erase(lean_object*, lean_object*);
extern lean_object* l_Lean_Meta_Simp_instInhabitedSimprocs_default;
extern lean_object* l_Lean_Meta_instInhabitedSimpTheorems_default;
extern lean_object* lp_aesop_Aesop_instInhabitedBaseRuleSet_default;
lean_object* l_Lean_ScopedEnvExtension_getState___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_ScopedEnvExtension_modifyState___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_setEnv___redArg(lean_object*, lean_object*);
lean_object* l_Std_DHashMap_Internal_AssocList_foldlM___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
lean_object* l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*);
lean_object* l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(lean_object*, uint8_t);
lean_object* l_Id_instMonad___lam__6(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__3(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__2___boxed(lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
size_t lean_array_size(lean_object*);
lean_object* l___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*);
lean_object* lean_array_to_list(lean_object*);
lean_object* l_List_mapTR_loop___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MessageData_ofList(lean_object*);
lean_object* l_Lean_ScopedEnvExtension_add___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t);
lean_object* l_Lean_PersistentHashMap_erase___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*);
lean_object* lp_aesop_Aesop_getDeclaredRuleSets();
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* lean_array_uget(lean_object*, size_t);
lean_object* l_Lean_Meta_getSimpExtension_x3f(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_Simp_getSimprocExtension_x3f(lean_object*);
lean_object* lean_st_ref_get(lean_object*);
lean_object* l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lean_io_error_to_string(lean_object*);
lean_object* l_Lean_MessageData_ofFormat(lean_object*);
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
size_t lean_usize_add(size_t, size_t);
uint8_t lean_usize_dec_eq(size_t, size_t);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_BaseRuleSet_add(lean_object*, lean_object*);
extern lean_object* lp_aesop_Aesop_BaseRuleSet_empty;
lean_object* l_Lean_registerSimpleScopedEnvExtension___redArg(lean_object*);
lean_object* lean_string_append(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Lean_Meta_registerSimpAttr(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_Simp_registerSimprocAttr(lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* lp_aesop_Aesop_declaredRuleSetsRef;
lean_object* lean_st_ref_take(lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* lean_nat_mul(lean_object*, lean_object*);
lean_object* lean_nat_div(lean_object*, lean_object*);
lean_object* lean_mk_array(lean_object*, lean_object*);
lean_object* lean_array_fget(lean_object*, lean_object*);
lean_object* lean_array_fset(lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_GlobalRuleSetMember_name(lean_object*);
uint8_t lp_aesop_Aesop_GlobalRuleSet_contains(lean_object*, lean_object*);
extern lean_object* lp_aesop_Aesop_builtinRuleSetNames;
lean_object* lp_aesop_Aesop_getDefaultRuleSetNames();
lean_object* l_Lean_Meta_instBEqOrigin___lam__0___boxed(lean_object*, lean_object*);
lean_object* l_Lean_Meta_instHashableOrigin___lam__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_extensionDescr___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_extensionDescr___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_extensionDescr___lam__1(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_extensionDescr___lam__1___boxed(lean_object*);
static const lean_closure_object lp_aesop_Aesop_Frontend_extensionDescr___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_BaseRuleSet_add, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_Frontend_extensionDescr___closed__0 = (const lean_object*)&lp_aesop_Aesop_Frontend_extensionDescr___closed__0_value;
static const lean_closure_object lp_aesop_Aesop_Frontend_extensionDescr___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_Frontend_extensionDescr___lam__0___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_Frontend_extensionDescr___closed__1 = (const lean_object*)&lp_aesop_Aesop_Frontend_extensionDescr___closed__1_value;
static const lean_closure_object lp_aesop_Aesop_Frontend_extensionDescr___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_Frontend_extensionDescr___lam__1___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_Frontend_extensionDescr___closed__2 = (const lean_object*)&lp_aesop_Aesop_Frontend_extensionDescr___closed__2_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_extensionDescr(lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_Frontend_declareRuleSetUnchecked_spec__0_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_Frontend_declareRuleSetUnchecked_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_Frontend_declareRuleSetUnchecked_spec__0_spec__1_spec__2_spec__4___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_Frontend_declareRuleSetUnchecked_spec__0_spec__1_spec__2___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_Frontend_declareRuleSetUnchecked_spec__0_spec__1___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_Frontend_declareRuleSetUnchecked_spec__0_spec__2___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_Frontend_declareRuleSetUnchecked_spec__0___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Aesop_Frontend_declareRuleSetUnchecked_spec__1___redArg(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop_Aesop_Frontend_declareRuleSetUnchecked___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "aesop_"};
static const lean_object* lp_aesop_Aesop_Frontend_declareRuleSetUnchecked___closed__0 = (const lean_object*)&lp_aesop_Aesop_Frontend_declareRuleSetUnchecked___closed__0_value;
static const lean_string_object lp_aesop_Aesop_Frontend_declareRuleSetUnchecked___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 38, .m_capacity = 38, .m_length = 37, .m_data = "simp theorems in the Aesop rule set '"};
static const lean_object* lp_aesop_Aesop_Frontend_declareRuleSetUnchecked___closed__1 = (const lean_object*)&lp_aesop_Aesop_Frontend_declareRuleSetUnchecked___closed__1_value;
static const lean_string_object lp_aesop_Aesop_Frontend_declareRuleSetUnchecked___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "'"};
static const lean_object* lp_aesop_Aesop_Frontend_declareRuleSetUnchecked___closed__2 = (const lean_object*)&lp_aesop_Aesop_Frontend_declareRuleSetUnchecked___closed__2_value;
static const lean_string_object lp_aesop_Aesop_Frontend_declareRuleSetUnchecked___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "_proc"};
static const lean_object* lp_aesop_Aesop_Frontend_declareRuleSetUnchecked___closed__3 = (const lean_object*)&lp_aesop_Aesop_Frontend_declareRuleSetUnchecked___closed__3_value;
static const lean_string_object lp_aesop_Aesop_Frontend_declareRuleSetUnchecked___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 33, .m_capacity = 33, .m_length = 32, .m_data = "simprocs in the Aesop rule set '"};
static const lean_object* lp_aesop_Aesop_Frontend_declareRuleSetUnchecked___closed__4 = (const lean_object*)&lp_aesop_Aesop_Frontend_declareRuleSetUnchecked___closed__4_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_declareRuleSetUnchecked(lean_object*, uint8_t);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_declareRuleSetUnchecked___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_Frontend_declareRuleSetUnchecked_spec__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Aesop_Frontend_declareRuleSetUnchecked_spec__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_Frontend_declareRuleSetUnchecked_spec__0_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_Frontend_declareRuleSetUnchecked_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_Frontend_declareRuleSetUnchecked_spec__0_spec__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_Frontend_declareRuleSetUnchecked_spec__0_spec__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_Frontend_declareRuleSetUnchecked_spec__0_spec__1_spec__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_Frontend_declareRuleSetUnchecked_spec__0_spec__1_spec__2_spec__4(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Std_DHashMap_Internal_Raw_u2080_contains___at___00Aesop_Frontend_isRuleSetDeclared_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_contains___at___00Aesop_Frontend_isRuleSetDeclared_spec__0___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_isRuleSetDeclared(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_isRuleSetDeclared___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Std_DHashMap_Internal_Raw_u2080_contains___at___00Aesop_Frontend_isRuleSetDeclared_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_contains___at___00Aesop_Frontend_isRuleSetDeclared_spec__0___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop_Aesop_Frontend_checkRuleSetNotDeclared___redArg___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "rule set '"};
static const lean_object* lp_aesop_Aesop_Frontend_checkRuleSetNotDeclared___redArg___lam__0___closed__0 = (const lean_object*)&lp_aesop_Aesop_Frontend_checkRuleSetNotDeclared___redArg___lam__0___closed__0_value;
static lean_once_cell_t lp_aesop_Aesop_Frontend_checkRuleSetNotDeclared___redArg___lam__0___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Frontend_checkRuleSetNotDeclared___redArg___lam__0___closed__1;
static const lean_string_object lp_aesop_Aesop_Frontend_checkRuleSetNotDeclared___redArg___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "' already exists"};
static const lean_object* lp_aesop_Aesop_Frontend_checkRuleSetNotDeclared___redArg___lam__0___closed__2 = (const lean_object*)&lp_aesop_Aesop_Frontend_checkRuleSetNotDeclared___redArg___lam__0___closed__2_value;
static lean_once_cell_t lp_aesop_Aesop_Frontend_checkRuleSetNotDeclared___redArg___lam__0___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Frontend_checkRuleSetNotDeclared___redArg___lam__0___closed__3;
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_checkRuleSetNotDeclared___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_checkRuleSetNotDeclared___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_checkRuleSetNotDeclared___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_checkRuleSetNotDeclared(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_declareRuleSet___redArg___lam__0(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_declareRuleSet___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_declareRuleSet___redArg(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_declareRuleSet___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_declareRuleSet(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_declareRuleSet___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Aesop_Frontend_Extension_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Extension_3818590508____hygCtx___hyg_2__spec__0(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Aesop_Frontend_Extension_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Extension_3818590508____hygCtx___hyg_2__spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_aesop___private_Aesop_Frontend_Extension_0__Aesop_Frontend_initFn___closed__0_00___x40_Aesop_Frontend_Extension_3818590508____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Frontend_Extension_0__Aesop_Frontend_initFn___closed__0_00___x40_Aesop_Frontend_Extension_3818590508____hygCtx___hyg_2_;
static lean_once_cell_t lp_aesop___private_Aesop_Frontend_Extension_0__Aesop_Frontend_initFn___closed__1_00___x40_Aesop_Frontend_Extension_3818590508____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static uint8_t lp_aesop___private_Aesop_Frontend_Extension_0__Aesop_Frontend_initFn___closed__1_00___x40_Aesop_Frontend_Extension_3818590508____hygCtx___hyg_2_;
static lean_once_cell_t lp_aesop___private_Aesop_Frontend_Extension_0__Aesop_Frontend_initFn___closed__2_00___x40_Aesop_Frontend_Extension_3818590508____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static uint8_t lp_aesop___private_Aesop_Frontend_Extension_0__Aesop_Frontend_initFn___closed__2_00___x40_Aesop_Frontend_Extension_3818590508____hygCtx___hyg_2_;
static lean_once_cell_t lp_aesop___private_Aesop_Frontend_Extension_0__Aesop_Frontend_initFn___closed__3_00___x40_Aesop_Frontend_Extension_3818590508____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static size_t lp_aesop___private_Aesop_Frontend_Extension_0__Aesop_Frontend_initFn___closed__3_00___x40_Aesop_Frontend_Extension_3818590508____hygCtx___hyg_2_;
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Frontend_Extension_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Extension_3818590508____hygCtx___hyg_2_();
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Frontend_Extension_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Extension_3818590508____hygCtx___hyg_2____boxed(lean_object*);
static const lean_string_object lp_aesop_Aesop_Frontend_getGlobalRuleSetData___redArg___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 27, .m_capacity = 27, .m_length = 26, .m_data = "internal error: expected '"};
static const lean_object* lp_aesop_Aesop_Frontend_getGlobalRuleSetData___redArg___lam__0___closed__0 = (const lean_object*)&lp_aesop_Aesop_Frontend_getGlobalRuleSetData___redArg___lam__0___closed__0_value;
static lean_once_cell_t lp_aesop_Aesop_Frontend_getGlobalRuleSetData___redArg___lam__0___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Frontend_getGlobalRuleSetData___redArg___lam__0___closed__1;
static const lean_string_object lp_aesop_Aesop_Frontend_getGlobalRuleSetData___redArg___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 34, .m_capacity = 34, .m_length = 33, .m_data = "' to be a declared simp extension"};
static const lean_object* lp_aesop_Aesop_Frontend_getGlobalRuleSetData___redArg___lam__0___closed__2 = (const lean_object*)&lp_aesop_Aesop_Frontend_getGlobalRuleSetData___redArg___lam__0___closed__2_value;
static lean_once_cell_t lp_aesop_Aesop_Frontend_getGlobalRuleSetData___redArg___lam__0___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Frontend_getGlobalRuleSetData___redArg___lam__0___closed__3;
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_getGlobalRuleSetData___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_getGlobalRuleSetData___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_getGlobalRuleSetData___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop_Aesop_Frontend_getGlobalRuleSetData___redArg___lam__2___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "no such rule set: '"};
static const lean_object* lp_aesop_Aesop_Frontend_getGlobalRuleSetData___redArg___lam__2___closed__0 = (const lean_object*)&lp_aesop_Aesop_Frontend_getGlobalRuleSetData___redArg___lam__2___closed__0_value;
static lean_once_cell_t lp_aesop_Aesop_Frontend_getGlobalRuleSetData___redArg___lam__2___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Frontend_getGlobalRuleSetData___redArg___lam__2___closed__1;
static const lean_string_object lp_aesop_Aesop_Frontend_getGlobalRuleSetData___redArg___lam__2___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 177, .m_capacity = 177, .m_length = 176, .m_data = "'\n  (Use 'declare_aesop_rule_set' to declare rule sets.\n   Declared rule sets are not visible in the current file; they only become visible once you import the declaring file.)"};
static const lean_object* lp_aesop_Aesop_Frontend_getGlobalRuleSetData___redArg___lam__2___closed__2 = (const lean_object*)&lp_aesop_Aesop_Frontend_getGlobalRuleSetData___redArg___lam__2___closed__2_value;
static lean_once_cell_t lp_aesop_Aesop_Frontend_getGlobalRuleSetData___redArg___lam__2___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Frontend_getGlobalRuleSetData___redArg___lam__2___closed__3;
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_getGlobalRuleSetData___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_getGlobalRuleSetData___redArg___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_aesop_Aesop_Frontend_getGlobalRuleSetData___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Name_beq___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_Frontend_getGlobalRuleSetData___redArg___closed__0 = (const lean_object*)&lp_aesop_Aesop_Frontend_getGlobalRuleSetData___redArg___closed__0_value;
static const lean_closure_object lp_aesop_Aesop_Frontend_getGlobalRuleSetData___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Name_hash___override___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_Frontend_getGlobalRuleSetData___redArg___closed__1 = (const lean_object*)&lp_aesop_Aesop_Frontend_getGlobalRuleSetData___redArg___closed__1_value;
static const lean_closure_object lp_aesop_Aesop_Frontend_getGlobalRuleSetData___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_getDeclaredRuleSets___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_Frontend_getGlobalRuleSetData___redArg___closed__2 = (const lean_object*)&lp_aesop_Aesop_Frontend_getGlobalRuleSetData___redArg___closed__2_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_getGlobalRuleSetData___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_getGlobalRuleSetData(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_getGlobalRuleSetFromData___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_getGlobalRuleSetFromData___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_getGlobalRuleSetFromData___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_getGlobalRuleSetFromData(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_getGlobalRuleSetFromData___at___00Aesop_Frontend_getGlobalRuleSet_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_getGlobalRuleSetFromData___at___00Aesop_Frontend_getGlobalRuleSet_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_getGlobalRuleSetFromData___at___00Aesop_Frontend_getGlobalRuleSet_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_getGlobalRuleSetFromData___at___00Aesop_Frontend_getGlobalRuleSet_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_getGlobalRuleSet_spec__0_spec__0_spec__2___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_getGlobalRuleSet_spec__0_spec__0_spec__2___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_getGlobalRuleSet_spec__0_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_getGlobalRuleSet_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*);
static lean_once_cell_t lp_aesop_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_getGlobalRuleSet_spec__0_spec__1_spec__4___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_getGlobalRuleSet_spec__0_spec__1_spec__4___closed__0;
static lean_once_cell_t lp_aesop_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_getGlobalRuleSet_spec__0_spec__1_spec__4___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_getGlobalRuleSet_spec__0_spec__1_spec__4___closed__1;
static lean_once_cell_t lp_aesop_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_getGlobalRuleSet_spec__0_spec__1_spec__4___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_getGlobalRuleSet_spec__0_spec__1_spec__4___closed__2;
static lean_once_cell_t lp_aesop_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_getGlobalRuleSet_spec__0_spec__1_spec__4___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_getGlobalRuleSet_spec__0_spec__1_spec__4___closed__3;
static lean_once_cell_t lp_aesop_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_getGlobalRuleSet_spec__0_spec__1_spec__4___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_getGlobalRuleSet_spec__0_spec__1_spec__4___closed__4;
static lean_once_cell_t lp_aesop_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_getGlobalRuleSet_spec__0_spec__1_spec__4___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_getGlobalRuleSet_spec__0_spec__1_spec__4___closed__5;
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_getGlobalRuleSet_spec__0_spec__1_spec__4(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_getGlobalRuleSet_spec__0_spec__1_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_getGlobalRuleSet_spec__0_spec__1___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_getGlobalRuleSet_spec__0_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_getGlobalRuleSet_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_getGlobalRuleSet_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_getGlobalRuleSet(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_getGlobalRuleSet___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_getGlobalRuleSet_spec__0_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_getGlobalRuleSet_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_getGlobalRuleSet_spec__0_spec__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_getGlobalRuleSet_spec__0_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_getGlobalRuleSet_spec__0_spec__0_spec__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_getGlobalRuleSet_spec__0_spec__0_spec__2___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Frontend_getGlobalRuleSets_spec__0(size_t, size_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Frontend_getGlobalRuleSets_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_getGlobalRuleSets(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_getGlobalRuleSets___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00Aesop_Frontend_getDefaultGlobalRuleSets_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Frontend_getDefaultGlobalRuleSets_spec__1(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Frontend_getDefaultGlobalRuleSets_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_getDefaultGlobalRuleSets(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_getDefaultGlobalRuleSets___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00Aesop_Frontend_getDeclaredGlobalRuleSets_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00Aesop_Frontend_getDeclaredGlobalRuleSets_spec__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Frontend_getDeclaredGlobalRuleSets_spec__2(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Frontend_getDeclaredGlobalRuleSets_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Frontend_getDeclaredGlobalRuleSets_spec__1(size_t, size_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Frontend_getDeclaredGlobalRuleSets_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_getDeclaredGlobalRuleSets(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_getDeclaredGlobalRuleSets___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_modifyGetGlobalRuleSet___redArg___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_modifyGetGlobalRuleSet___redArg___lam__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_modifyGetGlobalRuleSet___redArg___lam__1(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_modifyGetGlobalRuleSet___redArg___lam__1___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_modifyGetGlobalRuleSet___redArg___lam__2(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_modifyGetGlobalRuleSet___redArg___lam__2___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_modifyGetGlobalRuleSet___redArg___lam__3(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_modifyGetGlobalRuleSet___redArg___lam__3___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_modifyGetGlobalRuleSet___redArg___lam__4(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_modifyGetGlobalRuleSet___redArg___lam__4___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_modifyGetGlobalRuleSet___redArg___lam__5(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_modifyGetGlobalRuleSet___redArg___lam__6(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_modifyGetGlobalRuleSet___redArg___lam__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_modifyGetGlobalRuleSet___redArg___lam__7(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_aesop_Aesop_Frontend_modifyGetGlobalRuleSet___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_Frontend_modifyGetGlobalRuleSet___redArg___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_Frontend_modifyGetGlobalRuleSet___redArg___closed__0 = (const lean_object*)&lp_aesop_Aesop_Frontend_modifyGetGlobalRuleSet___redArg___closed__0_value;
static const lean_closure_object lp_aesop_Aesop_Frontend_modifyGetGlobalRuleSet___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_Frontend_modifyGetGlobalRuleSet___redArg___lam__1___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_Frontend_modifyGetGlobalRuleSet___redArg___closed__1 = (const lean_object*)&lp_aesop_Aesop_Frontend_modifyGetGlobalRuleSet___redArg___closed__1_value;
static const lean_closure_object lp_aesop_Aesop_Frontend_modifyGetGlobalRuleSet___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_Frontend_modifyGetGlobalRuleSet___redArg___lam__2___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_Frontend_modifyGetGlobalRuleSet___redArg___closed__2 = (const lean_object*)&lp_aesop_Aesop_Frontend_modifyGetGlobalRuleSet___redArg___closed__2_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_modifyGetGlobalRuleSet___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_modifyGetGlobalRuleSet(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_modifyGlobalRuleSet___lam__0(lean_object*, lean_object*);
static lean_once_cell_t lp_aesop_Lean_setEnv___at___00Aesop_Frontend_modifyGetGlobalRuleSet___at___00Aesop_Frontend_modifyGlobalRuleSet_spec__0_spec__0___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Lean_setEnv___at___00Aesop_Frontend_modifyGetGlobalRuleSet___at___00Aesop_Frontend_modifyGlobalRuleSet_spec__0_spec__0___redArg___closed__0;
static lean_once_cell_t lp_aesop_Lean_setEnv___at___00Aesop_Frontend_modifyGetGlobalRuleSet___at___00Aesop_Frontend_modifyGlobalRuleSet_spec__0_spec__0___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Lean_setEnv___at___00Aesop_Frontend_modifyGetGlobalRuleSet___at___00Aesop_Frontend_modifyGlobalRuleSet_spec__0_spec__0___redArg___closed__1;
static lean_once_cell_t lp_aesop_Lean_setEnv___at___00Aesop_Frontend_modifyGetGlobalRuleSet___at___00Aesop_Frontend_modifyGlobalRuleSet_spec__0_spec__0___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Lean_setEnv___at___00Aesop_Frontend_modifyGetGlobalRuleSet___at___00Aesop_Frontend_modifyGlobalRuleSet_spec__0_spec__0___redArg___closed__2;
LEAN_EXPORT lean_object* lp_aesop_Lean_setEnv___at___00Aesop_Frontend_modifyGetGlobalRuleSet___at___00Aesop_Frontend_modifyGlobalRuleSet_spec__0_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_setEnv___at___00Aesop_Frontend_modifyGetGlobalRuleSet___at___00Aesop_Frontend_modifyGlobalRuleSet_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_modifyGetGlobalRuleSet___at___00Aesop_Frontend_modifyGlobalRuleSet_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_modifyGetGlobalRuleSet___at___00Aesop_Frontend_modifyGlobalRuleSet_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_modifyGlobalRuleSet(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_modifyGlobalRuleSet___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_setEnv___at___00Aesop_Frontend_modifyGetGlobalRuleSet___at___00Aesop_Frontend_modifyGlobalRuleSet_spec__0_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_setEnv___at___00Aesop_Frontend_modifyGetGlobalRuleSet___at___00Aesop_Frontend_modifyGlobalRuleSet_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_modifyGetGlobalRuleSet___at___00Aesop_Frontend_modifyGlobalRuleSet_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_modifyGetGlobalRuleSet___at___00Aesop_Frontend_modifyGlobalRuleSet_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_addGlobalRule___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_addGlobalRule___redArg___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_addGlobalRule___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_addGlobalRule___redArg___lam__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_addGlobalRule___redArg___lam__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_addGlobalRule___redArg___lam__5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_addGlobalRule___redArg___lam__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_addGlobalRule___redArg___lam__6(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_addGlobalRule___redArg___lam__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_addGlobalRule___redArg___lam__7(lean_object*, lean_object*);
static const lean_string_object lp_aesop_Aesop_Frontend_addGlobalRule___redArg___lam__8___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "aesop: rule '"};
static const lean_object* lp_aesop_Aesop_Frontend_addGlobalRule___redArg___lam__8___closed__0 = (const lean_object*)&lp_aesop_Aesop_Frontend_addGlobalRule___redArg___lam__8___closed__0_value;
static lean_once_cell_t lp_aesop_Aesop_Frontend_addGlobalRule___redArg___lam__8___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Frontend_addGlobalRule___redArg___lam__8___closed__1;
static const lean_string_object lp_aesop_Aesop_Frontend_addGlobalRule___redArg___lam__8___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 38, .m_capacity = 38, .m_length = 37, .m_data = "' is already registered in rule set '"};
static const lean_object* lp_aesop_Aesop_Frontend_addGlobalRule___redArg___lam__8___closed__2 = (const lean_object*)&lp_aesop_Aesop_Frontend_addGlobalRule___redArg___lam__8___closed__2_value;
static lean_once_cell_t lp_aesop_Aesop_Frontend_addGlobalRule___redArg___lam__8___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Frontend_addGlobalRule___redArg___lam__8___closed__3;
static lean_once_cell_t lp_aesop_Aesop_Frontend_addGlobalRule___redArg___lam__8___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Frontend_addGlobalRule___redArg___lam__8___closed__4;
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_addGlobalRule___redArg___lam__8(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_addGlobalRule___redArg___lam__8___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_addGlobalRule___redArg___lam__9(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_addGlobalRule___redArg___lam__9___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_aesop_Aesop_Frontend_addGlobalRule___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Meta_instBEqOrigin___lam__0___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_Frontend_addGlobalRule___redArg___closed__0 = (const lean_object*)&lp_aesop_Aesop_Frontend_addGlobalRule___redArg___closed__0_value;
static const lean_closure_object lp_aesop_Aesop_Frontend_addGlobalRule___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Meta_instHashableOrigin___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_Frontend_addGlobalRule___redArg___closed__1 = (const lean_object*)&lp_aesop_Aesop_Frontend_addGlobalRule___redArg___closed__1_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_addGlobalRule___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, uint8_t);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_addGlobalRule___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_addGlobalRule(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, uint8_t);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_addGlobalRule___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Frontend_Extension_0__Aesop_Frontend_eraseGlobalRules_go___redArg___lam__0(lean_object*, uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Frontend_Extension_0__Aesop_Frontend_eraseGlobalRules_go___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Frontend_Extension_0__Aesop_Frontend_eraseGlobalRules_go___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Frontend_Extension_0__Aesop_Frontend_eraseGlobalRules_go___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Frontend_Extension_0__Aesop_Frontend_eraseGlobalRules_go(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Frontend_Extension_0__Aesop_Frontend_eraseGlobalRules_go___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop_Aesop_Frontend_eraseGlobalRules___redArg___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 63, .m_capacity = 63, .m_length = 62, .m_data = "' is not registered (with the given features) in any rule set."};
static const lean_object* lp_aesop_Aesop_Frontend_eraseGlobalRules___redArg___lam__0___closed__0 = (const lean_object*)&lp_aesop_Aesop_Frontend_eraseGlobalRules___redArg___lam__0___closed__0_value;
static lean_once_cell_t lp_aesop_Aesop_Frontend_eraseGlobalRules___redArg___lam__0___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Frontend_eraseGlobalRules___redArg___lam__0___closed__1;
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_eraseGlobalRules___redArg___lam__0(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, uint8_t);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_eraseGlobalRules___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_eraseGlobalRules___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_eraseGlobalRules___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_eraseGlobalRules___redArg___lam__2(lean_object*, lean_object*, uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_eraseGlobalRules___redArg___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_eraseGlobalRules___redArg___lam__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_eraseGlobalRules___redArg___lam__4(uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_eraseGlobalRules___redArg___lam__4___boxed(lean_object*, lean_object*);
static const lean_string_object lp_aesop_Aesop_Frontend_eraseGlobalRules___redArg___lam__5___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 71, .m_capacity = 71, .m_length = 70, .m_data = "' is not registered (with the given features) in any of the rule sets "};
static const lean_object* lp_aesop_Aesop_Frontend_eraseGlobalRules___redArg___lam__5___closed__0 = (const lean_object*)&lp_aesop_Aesop_Frontend_eraseGlobalRules___redArg___lam__5___closed__0_value;
static lean_once_cell_t lp_aesop_Aesop_Frontend_eraseGlobalRules___redArg___lam__5___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Frontend_eraseGlobalRules___redArg___lam__5___closed__1;
static const lean_closure_object lp_aesop_Aesop_Frontend_eraseGlobalRules___redArg___lam__5___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__0, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_Frontend_eraseGlobalRules___redArg___lam__5___closed__2 = (const lean_object*)&lp_aesop_Aesop_Frontend_eraseGlobalRules___redArg___lam__5___closed__2_value;
static const lean_closure_object lp_aesop_Aesop_Frontend_eraseGlobalRules___redArg___lam__5___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__1___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_Frontend_eraseGlobalRules___redArg___lam__5___closed__3 = (const lean_object*)&lp_aesop_Aesop_Frontend_eraseGlobalRules___redArg___lam__5___closed__3_value;
static const lean_closure_object lp_aesop_Aesop_Frontend_eraseGlobalRules___redArg___lam__5___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__2___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_Frontend_eraseGlobalRules___redArg___lam__5___closed__4 = (const lean_object*)&lp_aesop_Aesop_Frontend_eraseGlobalRules___redArg___lam__5___closed__4_value;
static const lean_closure_object lp_aesop_Aesop_Frontend_eraseGlobalRules___redArg___lam__5___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__3, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_Frontend_eraseGlobalRules___redArg___lam__5___closed__5 = (const lean_object*)&lp_aesop_Aesop_Frontend_eraseGlobalRules___redArg___lam__5___closed__5_value;
static const lean_closure_object lp_aesop_Aesop_Frontend_eraseGlobalRules___redArg___lam__5___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__4___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_Frontend_eraseGlobalRules___redArg___lam__5___closed__6 = (const lean_object*)&lp_aesop_Aesop_Frontend_eraseGlobalRules___redArg___lam__5___closed__6_value;
static const lean_closure_object lp_aesop_Aesop_Frontend_eraseGlobalRules___redArg___lam__5___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__5___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_Frontend_eraseGlobalRules___redArg___lam__5___closed__7 = (const lean_object*)&lp_aesop_Aesop_Frontend_eraseGlobalRules___redArg___lam__5___closed__7_value;
static const lean_closure_object lp_aesop_Aesop_Frontend_eraseGlobalRules___redArg___lam__5___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__6, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_Frontend_eraseGlobalRules___redArg___lam__5___closed__8 = (const lean_object*)&lp_aesop_Aesop_Frontend_eraseGlobalRules___redArg___lam__5___closed__8_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_eraseGlobalRules___redArg___lam__5___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_eraseGlobalRules___redArg___lam__5___closed__2_value),((lean_object*)&lp_aesop_Aesop_Frontend_eraseGlobalRules___redArg___lam__5___closed__3_value)}};
static const lean_object* lp_aesop_Aesop_Frontend_eraseGlobalRules___redArg___lam__5___closed__9 = (const lean_object*)&lp_aesop_Aesop_Frontend_eraseGlobalRules___redArg___lam__5___closed__9_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_eraseGlobalRules___redArg___lam__5___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*5 + 0, .m_other = 5, .m_tag = 0}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_eraseGlobalRules___redArg___lam__5___closed__9_value),((lean_object*)&lp_aesop_Aesop_Frontend_eraseGlobalRules___redArg___lam__5___closed__4_value),((lean_object*)&lp_aesop_Aesop_Frontend_eraseGlobalRules___redArg___lam__5___closed__5_value),((lean_object*)&lp_aesop_Aesop_Frontend_eraseGlobalRules___redArg___lam__5___closed__6_value),((lean_object*)&lp_aesop_Aesop_Frontend_eraseGlobalRules___redArg___lam__5___closed__7_value)}};
static const lean_object* lp_aesop_Aesop_Frontend_eraseGlobalRules___redArg___lam__5___closed__10 = (const lean_object*)&lp_aesop_Aesop_Frontend_eraseGlobalRules___redArg___lam__5___closed__10_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_eraseGlobalRules___redArg___lam__5___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_eraseGlobalRules___redArg___lam__5___closed__10_value),((lean_object*)&lp_aesop_Aesop_Frontend_eraseGlobalRules___redArg___lam__5___closed__8_value)}};
static const lean_object* lp_aesop_Aesop_Frontend_eraseGlobalRules___redArg___lam__5___closed__11 = (const lean_object*)&lp_aesop_Aesop_Frontend_eraseGlobalRules___redArg___lam__5___closed__11_value;
static const lean_closure_object lp_aesop_Aesop_Frontend_eraseGlobalRules___redArg___lam__5___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_stringToMessageData, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_Frontend_eraseGlobalRules___redArg___lam__5___closed__12 = (const lean_object*)&lp_aesop_Aesop_Frontend_eraseGlobalRules___redArg___lam__5___closed__12_value;
static const lean_string_object lp_aesop_Aesop_Frontend_eraseGlobalRules___redArg___lam__5___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "."};
static const lean_object* lp_aesop_Aesop_Frontend_eraseGlobalRules___redArg___lam__5___closed__13 = (const lean_object*)&lp_aesop_Aesop_Frontend_eraseGlobalRules___redArg___lam__5___closed__13_value;
static lean_once_cell_t lp_aesop_Aesop_Frontend_eraseGlobalRules___redArg___lam__5___closed__14_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Frontend_eraseGlobalRules___redArg___lam__5___closed__14;
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_eraseGlobalRules___redArg___lam__5(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_eraseGlobalRules___redArg___lam__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_eraseGlobalRules___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_eraseGlobalRules___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_eraseGlobalRules(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_eraseGlobalRules___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_extensionDescr___lam__0(lean_object* v_x_1_, lean_object* v_a_2_){
_start:
{
lean_object* v___x_3_; lean_object* v___x_4_; 
v___x_3_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3_, 0, v_a_2_);
lean_inc_ref_n(v___x_3_, 2);
v___x_4_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_4_, 0, v___x_3_);
lean_ctor_set(v___x_4_, 1, v___x_3_);
lean_ctor_set(v___x_4_, 2, v___x_3_);
return v___x_4_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_extensionDescr___lam__0___boxed(lean_object* v_x_5_, lean_object* v_a_6_){
_start:
{
lean_object* v_res_7_; 
v_res_7_ = lp_aesop_Aesop_Frontend_extensionDescr___lam__0(v_x_5_, v_a_6_);
lean_dec_ref(v_x_5_);
return v_res_7_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_extensionDescr___lam__1(lean_object* v___y_8_){
_start:
{
lean_inc_ref(v___y_8_);
return v___y_8_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_extensionDescr___lam__1___boxed(lean_object* v___y_9_){
_start:
{
lean_object* v_res_10_; 
v_res_10_ = lp_aesop_Aesop_Frontend_extensionDescr___lam__1(v___y_9_);
lean_dec_ref(v___y_9_);
return v_res_10_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_extensionDescr(lean_object* v_rsName_14_){
_start:
{
lean_object* v___f_15_; lean_object* v___f_16_; lean_object* v___f_17_; lean_object* v___x_18_; lean_object* v___x_19_; 
v___f_15_ = ((lean_object*)(lp_aesop_Aesop_Frontend_extensionDescr___closed__0));
v___f_16_ = ((lean_object*)(lp_aesop_Aesop_Frontend_extensionDescr___closed__1));
v___f_17_ = ((lean_object*)(lp_aesop_Aesop_Frontend_extensionDescr___closed__2));
v___x_18_ = lp_aesop_Aesop_BaseRuleSet_empty;
v___x_19_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_19_, 0, v_rsName_14_);
lean_ctor_set(v___x_19_, 1, v___f_15_);
lean_ctor_set(v___x_19_, 2, v___x_18_);
lean_ctor_set(v___x_19_, 3, v___f_17_);
lean_ctor_set(v___x_19_, 4, v___f_16_);
return v___x_19_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_Frontend_declareRuleSetUnchecked_spec__0_spec__0___redArg(lean_object* v_a_20_, lean_object* v_x_21_){
_start:
{
if (lean_obj_tag(v_x_21_) == 0)
{
uint8_t v___x_22_; 
v___x_22_ = 0;
return v___x_22_;
}
else
{
lean_object* v_key_23_; lean_object* v_tail_24_; uint8_t v___x_25_; 
v_key_23_ = lean_ctor_get(v_x_21_, 0);
v_tail_24_ = lean_ctor_get(v_x_21_, 2);
v___x_25_ = lean_name_eq(v_key_23_, v_a_20_);
if (v___x_25_ == 0)
{
v_x_21_ = v_tail_24_;
goto _start;
}
else
{
return v___x_25_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_Frontend_declareRuleSetUnchecked_spec__0_spec__0___redArg___boxed(lean_object* v_a_27_, lean_object* v_x_28_){
_start:
{
uint8_t v_res_29_; lean_object* v_r_30_; 
v_res_29_ = lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_Frontend_declareRuleSetUnchecked_spec__0_spec__0___redArg(v_a_27_, v_x_28_);
lean_dec(v_x_28_);
lean_dec(v_a_27_);
v_r_30_ = lean_box(v_res_29_);
return v_r_30_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_Frontend_declareRuleSetUnchecked_spec__0_spec__1_spec__2_spec__4___redArg(lean_object* v_x_31_, lean_object* v_x_32_){
_start:
{
if (lean_obj_tag(v_x_32_) == 0)
{
return v_x_31_;
}
else
{
lean_object* v_key_33_; lean_object* v_value_34_; lean_object* v_tail_35_; lean_object* v___x_37_; uint8_t v_isShared_38_; uint8_t v_isSharedCheck_61_; 
v_key_33_ = lean_ctor_get(v_x_32_, 0);
v_value_34_ = lean_ctor_get(v_x_32_, 1);
v_tail_35_ = lean_ctor_get(v_x_32_, 2);
v_isSharedCheck_61_ = !lean_is_exclusive(v_x_32_);
if (v_isSharedCheck_61_ == 0)
{
v___x_37_ = v_x_32_;
v_isShared_38_ = v_isSharedCheck_61_;
goto v_resetjp_36_;
}
else
{
lean_inc(v_tail_35_);
lean_inc(v_value_34_);
lean_inc(v_key_33_);
lean_dec(v_x_32_);
v___x_37_ = lean_box(0);
v_isShared_38_ = v_isSharedCheck_61_;
goto v_resetjp_36_;
}
v_resetjp_36_:
{
lean_object* v___x_39_; uint64_t v___y_41_; 
v___x_39_ = lean_array_get_size(v_x_31_);
if (lean_obj_tag(v_key_33_) == 0)
{
uint64_t v___x_59_; 
v___x_59_ = 1723ULL;
v___y_41_ = v___x_59_;
goto v___jp_40_;
}
else
{
uint64_t v_hash_60_; 
v_hash_60_ = lean_ctor_get_uint64(v_key_33_, sizeof(void*)*2);
v___y_41_ = v_hash_60_;
goto v___jp_40_;
}
v___jp_40_:
{
uint64_t v___x_42_; uint64_t v___x_43_; uint64_t v_fold_44_; uint64_t v___x_45_; uint64_t v___x_46_; uint64_t v___x_47_; size_t v___x_48_; size_t v___x_49_; size_t v___x_50_; size_t v___x_51_; size_t v___x_52_; lean_object* v___x_53_; lean_object* v___x_55_; 
v___x_42_ = 32ULL;
v___x_43_ = lean_uint64_shift_right(v___y_41_, v___x_42_);
v_fold_44_ = lean_uint64_xor(v___y_41_, v___x_43_);
v___x_45_ = 16ULL;
v___x_46_ = lean_uint64_shift_right(v_fold_44_, v___x_45_);
v___x_47_ = lean_uint64_xor(v_fold_44_, v___x_46_);
v___x_48_ = lean_uint64_to_usize(v___x_47_);
v___x_49_ = lean_usize_of_nat(v___x_39_);
v___x_50_ = ((size_t)1ULL);
v___x_51_ = lean_usize_sub(v___x_49_, v___x_50_);
v___x_52_ = lean_usize_land(v___x_48_, v___x_51_);
v___x_53_ = lean_array_uget_borrowed(v_x_31_, v___x_52_);
lean_inc(v___x_53_);
if (v_isShared_38_ == 0)
{
lean_ctor_set(v___x_37_, 2, v___x_53_);
v___x_55_ = v___x_37_;
goto v_reusejp_54_;
}
else
{
lean_object* v_reuseFailAlloc_58_; 
v_reuseFailAlloc_58_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_58_, 0, v_key_33_);
lean_ctor_set(v_reuseFailAlloc_58_, 1, v_value_34_);
lean_ctor_set(v_reuseFailAlloc_58_, 2, v___x_53_);
v___x_55_ = v_reuseFailAlloc_58_;
goto v_reusejp_54_;
}
v_reusejp_54_:
{
lean_object* v___x_56_; 
v___x_56_ = lean_array_uset(v_x_31_, v___x_52_, v___x_55_);
v_x_31_ = v___x_56_;
v_x_32_ = v_tail_35_;
goto _start;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_Frontend_declareRuleSetUnchecked_spec__0_spec__1_spec__2___redArg(lean_object* v_i_62_, lean_object* v_source_63_, lean_object* v_target_64_){
_start:
{
lean_object* v___x_65_; uint8_t v___x_66_; 
v___x_65_ = lean_array_get_size(v_source_63_);
v___x_66_ = lean_nat_dec_lt(v_i_62_, v___x_65_);
if (v___x_66_ == 0)
{
lean_dec_ref(v_source_63_);
lean_dec(v_i_62_);
return v_target_64_;
}
else
{
lean_object* v_es_67_; lean_object* v___x_68_; lean_object* v_source_69_; lean_object* v_target_70_; lean_object* v___x_71_; lean_object* v___x_72_; 
v_es_67_ = lean_array_fget(v_source_63_, v_i_62_);
v___x_68_ = lean_box(0);
v_source_69_ = lean_array_fset(v_source_63_, v_i_62_, v___x_68_);
v_target_70_ = lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_Frontend_declareRuleSetUnchecked_spec__0_spec__1_spec__2_spec__4___redArg(v_target_64_, v_es_67_);
v___x_71_ = lean_unsigned_to_nat(1u);
v___x_72_ = lean_nat_add(v_i_62_, v___x_71_);
lean_dec(v_i_62_);
v_i_62_ = v___x_72_;
v_source_63_ = v_source_69_;
v_target_64_ = v_target_70_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_Frontend_declareRuleSetUnchecked_spec__0_spec__1___redArg(lean_object* v_data_74_){
_start:
{
lean_object* v___x_75_; lean_object* v___x_76_; lean_object* v_nbuckets_77_; lean_object* v___x_78_; lean_object* v___x_79_; lean_object* v___x_80_; lean_object* v___x_81_; 
v___x_75_ = lean_array_get_size(v_data_74_);
v___x_76_ = lean_unsigned_to_nat(2u);
v_nbuckets_77_ = lean_nat_mul(v___x_75_, v___x_76_);
v___x_78_ = lean_unsigned_to_nat(0u);
v___x_79_ = lean_box(0);
v___x_80_ = lean_mk_array(v_nbuckets_77_, v___x_79_);
v___x_81_ = lp_aesop___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_Frontend_declareRuleSetUnchecked_spec__0_spec__1_spec__2___redArg(v___x_78_, v_data_74_, v___x_80_);
return v___x_81_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_Frontend_declareRuleSetUnchecked_spec__0_spec__2___redArg(lean_object* v_a_82_, lean_object* v_b_83_, lean_object* v_x_84_){
_start:
{
if (lean_obj_tag(v_x_84_) == 0)
{
lean_dec(v_b_83_);
lean_dec(v_a_82_);
return v_x_84_;
}
else
{
lean_object* v_key_85_; lean_object* v_value_86_; lean_object* v_tail_87_; lean_object* v___x_89_; uint8_t v_isShared_90_; uint8_t v_isSharedCheck_99_; 
v_key_85_ = lean_ctor_get(v_x_84_, 0);
v_value_86_ = lean_ctor_get(v_x_84_, 1);
v_tail_87_ = lean_ctor_get(v_x_84_, 2);
v_isSharedCheck_99_ = !lean_is_exclusive(v_x_84_);
if (v_isSharedCheck_99_ == 0)
{
v___x_89_ = v_x_84_;
v_isShared_90_ = v_isSharedCheck_99_;
goto v_resetjp_88_;
}
else
{
lean_inc(v_tail_87_);
lean_inc(v_value_86_);
lean_inc(v_key_85_);
lean_dec(v_x_84_);
v___x_89_ = lean_box(0);
v_isShared_90_ = v_isSharedCheck_99_;
goto v_resetjp_88_;
}
v_resetjp_88_:
{
uint8_t v___x_91_; 
v___x_91_ = lean_name_eq(v_key_85_, v_a_82_);
if (v___x_91_ == 0)
{
lean_object* v___x_92_; lean_object* v___x_94_; 
v___x_92_ = lp_aesop_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_Frontend_declareRuleSetUnchecked_spec__0_spec__2___redArg(v_a_82_, v_b_83_, v_tail_87_);
if (v_isShared_90_ == 0)
{
lean_ctor_set(v___x_89_, 2, v___x_92_);
v___x_94_ = v___x_89_;
goto v_reusejp_93_;
}
else
{
lean_object* v_reuseFailAlloc_95_; 
v_reuseFailAlloc_95_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_95_, 0, v_key_85_);
lean_ctor_set(v_reuseFailAlloc_95_, 1, v_value_86_);
lean_ctor_set(v_reuseFailAlloc_95_, 2, v___x_92_);
v___x_94_ = v_reuseFailAlloc_95_;
goto v_reusejp_93_;
}
v_reusejp_93_:
{
return v___x_94_;
}
}
else
{
lean_object* v___x_97_; 
lean_dec(v_value_86_);
lean_dec(v_key_85_);
if (v_isShared_90_ == 0)
{
lean_ctor_set(v___x_89_, 1, v_b_83_);
lean_ctor_set(v___x_89_, 0, v_a_82_);
v___x_97_ = v___x_89_;
goto v_reusejp_96_;
}
else
{
lean_object* v_reuseFailAlloc_98_; 
v_reuseFailAlloc_98_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_98_, 0, v_a_82_);
lean_ctor_set(v_reuseFailAlloc_98_, 1, v_b_83_);
lean_ctor_set(v_reuseFailAlloc_98_, 2, v_tail_87_);
v___x_97_ = v_reuseFailAlloc_98_;
goto v_reusejp_96_;
}
v_reusejp_96_:
{
return v___x_97_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_Frontend_declareRuleSetUnchecked_spec__0___redArg(lean_object* v_m_100_, lean_object* v_a_101_, lean_object* v_b_102_){
_start:
{
lean_object* v_size_103_; lean_object* v_buckets_104_; lean_object* v___x_106_; uint8_t v_isShared_107_; uint8_t v_isSharedCheck_150_; 
v_size_103_ = lean_ctor_get(v_m_100_, 0);
v_buckets_104_ = lean_ctor_get(v_m_100_, 1);
v_isSharedCheck_150_ = !lean_is_exclusive(v_m_100_);
if (v_isSharedCheck_150_ == 0)
{
v___x_106_ = v_m_100_;
v_isShared_107_ = v_isSharedCheck_150_;
goto v_resetjp_105_;
}
else
{
lean_inc(v_buckets_104_);
lean_inc(v_size_103_);
lean_dec(v_m_100_);
v___x_106_ = lean_box(0);
v_isShared_107_ = v_isSharedCheck_150_;
goto v_resetjp_105_;
}
v_resetjp_105_:
{
lean_object* v___x_108_; uint64_t v___y_110_; 
v___x_108_ = lean_array_get_size(v_buckets_104_);
if (lean_obj_tag(v_a_101_) == 0)
{
uint64_t v___x_148_; 
v___x_148_ = 1723ULL;
v___y_110_ = v___x_148_;
goto v___jp_109_;
}
else
{
uint64_t v_hash_149_; 
v_hash_149_ = lean_ctor_get_uint64(v_a_101_, sizeof(void*)*2);
v___y_110_ = v_hash_149_;
goto v___jp_109_;
}
v___jp_109_:
{
uint64_t v___x_111_; uint64_t v___x_112_; uint64_t v_fold_113_; uint64_t v___x_114_; uint64_t v___x_115_; uint64_t v___x_116_; size_t v___x_117_; size_t v___x_118_; size_t v___x_119_; size_t v___x_120_; size_t v___x_121_; lean_object* v_bkt_122_; uint8_t v___x_123_; 
v___x_111_ = 32ULL;
v___x_112_ = lean_uint64_shift_right(v___y_110_, v___x_111_);
v_fold_113_ = lean_uint64_xor(v___y_110_, v___x_112_);
v___x_114_ = 16ULL;
v___x_115_ = lean_uint64_shift_right(v_fold_113_, v___x_114_);
v___x_116_ = lean_uint64_xor(v_fold_113_, v___x_115_);
v___x_117_ = lean_uint64_to_usize(v___x_116_);
v___x_118_ = lean_usize_of_nat(v___x_108_);
v___x_119_ = ((size_t)1ULL);
v___x_120_ = lean_usize_sub(v___x_118_, v___x_119_);
v___x_121_ = lean_usize_land(v___x_117_, v___x_120_);
v_bkt_122_ = lean_array_uget_borrowed(v_buckets_104_, v___x_121_);
v___x_123_ = lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_Frontend_declareRuleSetUnchecked_spec__0_spec__0___redArg(v_a_101_, v_bkt_122_);
if (v___x_123_ == 0)
{
lean_object* v___x_124_; lean_object* v_size_x27_125_; lean_object* v___x_126_; lean_object* v_buckets_x27_127_; lean_object* v___x_128_; lean_object* v___x_129_; lean_object* v___x_130_; lean_object* v___x_131_; lean_object* v___x_132_; uint8_t v___x_133_; 
v___x_124_ = lean_unsigned_to_nat(1u);
v_size_x27_125_ = lean_nat_add(v_size_103_, v___x_124_);
lean_dec(v_size_103_);
lean_inc(v_bkt_122_);
v___x_126_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_126_, 0, v_a_101_);
lean_ctor_set(v___x_126_, 1, v_b_102_);
lean_ctor_set(v___x_126_, 2, v_bkt_122_);
v_buckets_x27_127_ = lean_array_uset(v_buckets_104_, v___x_121_, v___x_126_);
v___x_128_ = lean_unsigned_to_nat(4u);
v___x_129_ = lean_nat_mul(v_size_x27_125_, v___x_128_);
v___x_130_ = lean_unsigned_to_nat(3u);
v___x_131_ = lean_nat_div(v___x_129_, v___x_130_);
lean_dec(v___x_129_);
v___x_132_ = lean_array_get_size(v_buckets_x27_127_);
v___x_133_ = lean_nat_dec_le(v___x_131_, v___x_132_);
lean_dec(v___x_131_);
if (v___x_133_ == 0)
{
lean_object* v_val_134_; lean_object* v___x_136_; 
v_val_134_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_Frontend_declareRuleSetUnchecked_spec__0_spec__1___redArg(v_buckets_x27_127_);
if (v_isShared_107_ == 0)
{
lean_ctor_set(v___x_106_, 1, v_val_134_);
lean_ctor_set(v___x_106_, 0, v_size_x27_125_);
v___x_136_ = v___x_106_;
goto v_reusejp_135_;
}
else
{
lean_object* v_reuseFailAlloc_137_; 
v_reuseFailAlloc_137_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_137_, 0, v_size_x27_125_);
lean_ctor_set(v_reuseFailAlloc_137_, 1, v_val_134_);
v___x_136_ = v_reuseFailAlloc_137_;
goto v_reusejp_135_;
}
v_reusejp_135_:
{
return v___x_136_;
}
}
else
{
lean_object* v___x_139_; 
if (v_isShared_107_ == 0)
{
lean_ctor_set(v___x_106_, 1, v_buckets_x27_127_);
lean_ctor_set(v___x_106_, 0, v_size_x27_125_);
v___x_139_ = v___x_106_;
goto v_reusejp_138_;
}
else
{
lean_object* v_reuseFailAlloc_140_; 
v_reuseFailAlloc_140_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_140_, 0, v_size_x27_125_);
lean_ctor_set(v_reuseFailAlloc_140_, 1, v_buckets_x27_127_);
v___x_139_ = v_reuseFailAlloc_140_;
goto v_reusejp_138_;
}
v_reusejp_138_:
{
return v___x_139_;
}
}
}
else
{
lean_object* v___x_141_; lean_object* v_buckets_x27_142_; lean_object* v___x_143_; lean_object* v___x_144_; lean_object* v___x_146_; 
lean_inc(v_bkt_122_);
v___x_141_ = lean_box(0);
v_buckets_x27_142_ = lean_array_uset(v_buckets_104_, v___x_121_, v___x_141_);
v___x_143_ = lp_aesop_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_Frontend_declareRuleSetUnchecked_spec__0_spec__2___redArg(v_a_101_, v_b_102_, v_bkt_122_);
v___x_144_ = lean_array_uset(v_buckets_x27_142_, v___x_121_, v___x_143_);
if (v_isShared_107_ == 0)
{
lean_ctor_set(v___x_106_, 1, v___x_144_);
v___x_146_ = v___x_106_;
goto v_reusejp_145_;
}
else
{
lean_object* v_reuseFailAlloc_147_; 
v_reuseFailAlloc_147_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_147_, 0, v_size_103_);
lean_ctor_set(v_reuseFailAlloc_147_, 1, v___x_144_);
v___x_146_ = v_reuseFailAlloc_147_;
goto v_reusejp_145_;
}
v_reusejp_145_:
{
return v___x_146_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Aesop_Frontend_declareRuleSetUnchecked_spec__1___redArg(lean_object* v_m_151_, lean_object* v_a_152_, lean_object* v_b_153_){
_start:
{
lean_object* v_size_154_; lean_object* v_buckets_155_; lean_object* v___x_156_; uint64_t v___y_158_; 
v_size_154_ = lean_ctor_get(v_m_151_, 0);
v_buckets_155_ = lean_ctor_get(v_m_151_, 1);
v___x_156_ = lean_array_get_size(v_buckets_155_);
if (lean_obj_tag(v_a_152_) == 0)
{
uint64_t v___x_195_; 
v___x_195_ = 1723ULL;
v___y_158_ = v___x_195_;
goto v___jp_157_;
}
else
{
uint64_t v_hash_196_; 
v_hash_196_ = lean_ctor_get_uint64(v_a_152_, sizeof(void*)*2);
v___y_158_ = v_hash_196_;
goto v___jp_157_;
}
v___jp_157_:
{
uint64_t v___x_159_; uint64_t v___x_160_; uint64_t v_fold_161_; uint64_t v___x_162_; uint64_t v___x_163_; uint64_t v___x_164_; size_t v___x_165_; size_t v___x_166_; size_t v___x_167_; size_t v___x_168_; size_t v___x_169_; lean_object* v_bkt_170_; uint8_t v___x_171_; 
v___x_159_ = 32ULL;
v___x_160_ = lean_uint64_shift_right(v___y_158_, v___x_159_);
v_fold_161_ = lean_uint64_xor(v___y_158_, v___x_160_);
v___x_162_ = 16ULL;
v___x_163_ = lean_uint64_shift_right(v_fold_161_, v___x_162_);
v___x_164_ = lean_uint64_xor(v_fold_161_, v___x_163_);
v___x_165_ = lean_uint64_to_usize(v___x_164_);
v___x_166_ = lean_usize_of_nat(v___x_156_);
v___x_167_ = ((size_t)1ULL);
v___x_168_ = lean_usize_sub(v___x_166_, v___x_167_);
v___x_169_ = lean_usize_land(v___x_165_, v___x_168_);
v_bkt_170_ = lean_array_uget_borrowed(v_buckets_155_, v___x_169_);
v___x_171_ = lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_Frontend_declareRuleSetUnchecked_spec__0_spec__0___redArg(v_a_152_, v_bkt_170_);
if (v___x_171_ == 0)
{
lean_object* v___x_173_; uint8_t v_isShared_174_; uint8_t v_isSharedCheck_192_; 
lean_inc_ref(v_buckets_155_);
lean_inc(v_size_154_);
v_isSharedCheck_192_ = !lean_is_exclusive(v_m_151_);
if (v_isSharedCheck_192_ == 0)
{
lean_object* v_unused_193_; lean_object* v_unused_194_; 
v_unused_193_ = lean_ctor_get(v_m_151_, 1);
lean_dec(v_unused_193_);
v_unused_194_ = lean_ctor_get(v_m_151_, 0);
lean_dec(v_unused_194_);
v___x_173_ = v_m_151_;
v_isShared_174_ = v_isSharedCheck_192_;
goto v_resetjp_172_;
}
else
{
lean_dec(v_m_151_);
v___x_173_ = lean_box(0);
v_isShared_174_ = v_isSharedCheck_192_;
goto v_resetjp_172_;
}
v_resetjp_172_:
{
lean_object* v___x_175_; lean_object* v_size_x27_176_; lean_object* v___x_177_; lean_object* v_buckets_x27_178_; lean_object* v___x_179_; lean_object* v___x_180_; lean_object* v___x_181_; lean_object* v___x_182_; lean_object* v___x_183_; uint8_t v___x_184_; 
v___x_175_ = lean_unsigned_to_nat(1u);
v_size_x27_176_ = lean_nat_add(v_size_154_, v___x_175_);
lean_dec(v_size_154_);
lean_inc(v_bkt_170_);
v___x_177_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_177_, 0, v_a_152_);
lean_ctor_set(v___x_177_, 1, v_b_153_);
lean_ctor_set(v___x_177_, 2, v_bkt_170_);
v_buckets_x27_178_ = lean_array_uset(v_buckets_155_, v___x_169_, v___x_177_);
v___x_179_ = lean_unsigned_to_nat(4u);
v___x_180_ = lean_nat_mul(v_size_x27_176_, v___x_179_);
v___x_181_ = lean_unsigned_to_nat(3u);
v___x_182_ = lean_nat_div(v___x_180_, v___x_181_);
lean_dec(v___x_180_);
v___x_183_ = lean_array_get_size(v_buckets_x27_178_);
v___x_184_ = lean_nat_dec_le(v___x_182_, v___x_183_);
lean_dec(v___x_182_);
if (v___x_184_ == 0)
{
lean_object* v_val_185_; lean_object* v___x_187_; 
v_val_185_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_Frontend_declareRuleSetUnchecked_spec__0_spec__1___redArg(v_buckets_x27_178_);
if (v_isShared_174_ == 0)
{
lean_ctor_set(v___x_173_, 1, v_val_185_);
lean_ctor_set(v___x_173_, 0, v_size_x27_176_);
v___x_187_ = v___x_173_;
goto v_reusejp_186_;
}
else
{
lean_object* v_reuseFailAlloc_188_; 
v_reuseFailAlloc_188_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_188_, 0, v_size_x27_176_);
lean_ctor_set(v_reuseFailAlloc_188_, 1, v_val_185_);
v___x_187_ = v_reuseFailAlloc_188_;
goto v_reusejp_186_;
}
v_reusejp_186_:
{
return v___x_187_;
}
}
else
{
lean_object* v___x_190_; 
if (v_isShared_174_ == 0)
{
lean_ctor_set(v___x_173_, 1, v_buckets_x27_178_);
lean_ctor_set(v___x_173_, 0, v_size_x27_176_);
v___x_190_ = v___x_173_;
goto v_reusejp_189_;
}
else
{
lean_object* v_reuseFailAlloc_191_; 
v_reuseFailAlloc_191_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_191_, 0, v_size_x27_176_);
lean_ctor_set(v_reuseFailAlloc_191_, 1, v_buckets_x27_178_);
v___x_190_ = v_reuseFailAlloc_191_;
goto v_reusejp_189_;
}
v_reusejp_189_:
{
return v___x_190_;
}
}
}
}
else
{
lean_dec(v_b_153_);
lean_dec(v_a_152_);
return v_m_151_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_declareRuleSetUnchecked(lean_object* v_rsName_202_, uint8_t v_isDefault_203_){
_start:
{
lean_object* v___x_205_; lean_object* v___x_206_; 
lean_inc(v_rsName_202_);
v___x_205_ = lp_aesop_Aesop_Frontend_extensionDescr(v_rsName_202_);
v___x_206_ = l_Lean_registerSimpleScopedEnvExtension___redArg(v___x_205_);
if (lean_obj_tag(v___x_206_) == 0)
{
lean_object* v_a_207_; lean_object* v___x_208_; uint8_t v___x_209_; lean_object* v___x_210_; lean_object* v___x_211_; lean_object* v___x_212_; lean_object* v___x_213_; lean_object* v___x_214_; lean_object* v___x_215_; lean_object* v___x_216_; lean_object* v___x_217_; 
v_a_207_ = lean_ctor_get(v___x_206_, 0);
lean_inc(v_a_207_);
lean_dec_ref_known(v___x_206_, 1);
v___x_208_ = ((lean_object*)(lp_aesop_Aesop_Frontend_declareRuleSetUnchecked___closed__0));
v___x_209_ = 1;
lean_inc(v_rsName_202_);
v___x_210_ = l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(v_rsName_202_, v___x_209_);
v___x_211_ = lean_string_append(v___x_208_, v___x_210_);
lean_inc_ref(v___x_211_);
v___x_212_ = l_Lean_Name_mkStr1(v___x_211_);
v___x_213_ = ((lean_object*)(lp_aesop_Aesop_Frontend_declareRuleSetUnchecked___closed__1));
v___x_214_ = lean_string_append(v___x_213_, v___x_210_);
v___x_215_ = ((lean_object*)(lp_aesop_Aesop_Frontend_declareRuleSetUnchecked___closed__2));
v___x_216_ = lean_string_append(v___x_214_, v___x_215_);
lean_inc_n(v___x_212_, 2);
v___x_217_ = l_Lean_Meta_registerSimpAttr(v___x_212_, v___x_216_, v___x_212_);
if (lean_obj_tag(v___x_217_) == 0)
{
lean_object* v___x_218_; lean_object* v___x_219_; lean_object* v___x_220_; lean_object* v___x_221_; lean_object* v___x_222_; lean_object* v___x_223_; lean_object* v___x_224_; lean_object* v___x_225_; 
lean_dec_ref_known(v___x_217_, 1);
v___x_218_ = ((lean_object*)(lp_aesop_Aesop_Frontend_declareRuleSetUnchecked___closed__3));
v___x_219_ = lean_string_append(v___x_211_, v___x_218_);
v___x_220_ = l_Lean_Name_mkStr1(v___x_219_);
v___x_221_ = ((lean_object*)(lp_aesop_Aesop_Frontend_declareRuleSetUnchecked___closed__4));
v___x_222_ = lean_string_append(v___x_221_, v___x_210_);
lean_dec_ref(v___x_210_);
v___x_223_ = lean_string_append(v___x_222_, v___x_215_);
v___x_224_ = lean_box(0);
lean_inc_n(v___x_220_, 2);
v___x_225_ = l_Lean_Meta_Simp_registerSimprocAttr(v___x_220_, v___x_223_, v___x_224_, v___x_220_);
if (lean_obj_tag(v___x_225_) == 0)
{
lean_object* v___x_227_; uint8_t v_isShared_228_; uint8_t v_isSharedCheck_254_; 
v_isSharedCheck_254_ = !lean_is_exclusive(v___x_225_);
if (v_isSharedCheck_254_ == 0)
{
lean_object* v_unused_255_; 
v_unused_255_ = lean_ctor_get(v___x_225_, 0);
lean_dec(v_unused_255_);
v___x_227_ = v___x_225_;
v_isShared_228_ = v_isSharedCheck_254_;
goto v_resetjp_226_;
}
else
{
lean_dec(v___x_225_);
v___x_227_ = lean_box(0);
v_isShared_228_ = v_isSharedCheck_254_;
goto v_resetjp_226_;
}
v_resetjp_226_:
{
lean_object* v___x_229_; lean_object* v___x_230_; lean_object* v___y_232_; lean_object* v_ruleSets_237_; lean_object* v_defaultRuleSets_238_; lean_object* v___x_240_; uint8_t v_isShared_241_; uint8_t v_isSharedCheck_253_; 
v___x_229_ = lp_aesop_Aesop_declaredRuleSetsRef;
v___x_230_ = lean_st_ref_take(v___x_229_);
v_ruleSets_237_ = lean_ctor_get(v___x_230_, 0);
v_defaultRuleSets_238_ = lean_ctor_get(v___x_230_, 1);
v_isSharedCheck_253_ = !lean_is_exclusive(v___x_230_);
if (v_isSharedCheck_253_ == 0)
{
v___x_240_ = v___x_230_;
v_isShared_241_ = v_isSharedCheck_253_;
goto v_resetjp_239_;
}
else
{
lean_inc(v_defaultRuleSets_238_);
lean_inc(v_ruleSets_237_);
lean_dec(v___x_230_);
v___x_240_ = lean_box(0);
v_isShared_241_ = v_isSharedCheck_253_;
goto v_resetjp_239_;
}
v___jp_231_:
{
lean_object* v___x_233_; lean_object* v___x_235_; 
v___x_233_ = lean_st_ref_set(v___x_229_, v___y_232_);
if (v_isShared_228_ == 0)
{
lean_ctor_set(v___x_227_, 0, v___x_233_);
v___x_235_ = v___x_227_;
goto v_reusejp_234_;
}
else
{
lean_object* v_reuseFailAlloc_236_; 
v_reuseFailAlloc_236_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_236_, 0, v___x_233_);
v___x_235_ = v_reuseFailAlloc_236_;
goto v_reusejp_234_;
}
v_reusejp_234_:
{
return v___x_235_;
}
}
v_resetjp_239_:
{
lean_object* v___x_242_; lean_object* v___x_243_; lean_object* v___x_244_; 
v___x_242_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_242_, 0, v___x_212_);
lean_ctor_set(v___x_242_, 1, v___x_220_);
v___x_243_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_243_, 0, v_a_207_);
lean_ctor_set(v___x_243_, 1, v___x_242_);
lean_inc(v_rsName_202_);
v___x_244_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_Frontend_declareRuleSetUnchecked_spec__0___redArg(v_ruleSets_237_, v_rsName_202_, v___x_243_);
if (v_isDefault_203_ == 0)
{
lean_object* v___x_246_; 
lean_dec(v_rsName_202_);
if (v_isShared_241_ == 0)
{
lean_ctor_set(v___x_240_, 0, v___x_244_);
v___x_246_ = v___x_240_;
goto v_reusejp_245_;
}
else
{
lean_object* v_reuseFailAlloc_247_; 
v_reuseFailAlloc_247_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_247_, 0, v___x_244_);
lean_ctor_set(v_reuseFailAlloc_247_, 1, v_defaultRuleSets_238_);
v___x_246_ = v_reuseFailAlloc_247_;
goto v_reusejp_245_;
}
v_reusejp_245_:
{
v___y_232_ = v___x_246_;
goto v___jp_231_;
}
}
else
{
lean_object* v___x_248_; lean_object* v___x_249_; lean_object* v___x_251_; 
v___x_248_ = lean_box(0);
v___x_249_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Aesop_Frontend_declareRuleSetUnchecked_spec__1___redArg(v_defaultRuleSets_238_, v_rsName_202_, v___x_248_);
if (v_isShared_241_ == 0)
{
lean_ctor_set(v___x_240_, 1, v___x_249_);
lean_ctor_set(v___x_240_, 0, v___x_244_);
v___x_251_ = v___x_240_;
goto v_reusejp_250_;
}
else
{
lean_object* v_reuseFailAlloc_252_; 
v_reuseFailAlloc_252_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_252_, 0, v___x_244_);
lean_ctor_set(v_reuseFailAlloc_252_, 1, v___x_249_);
v___x_251_ = v_reuseFailAlloc_252_;
goto v_reusejp_250_;
}
v_reusejp_250_:
{
v___y_232_ = v___x_251_;
goto v___jp_231_;
}
}
}
}
}
else
{
lean_object* v_a_256_; lean_object* v___x_258_; uint8_t v_isShared_259_; uint8_t v_isSharedCheck_263_; 
lean_dec(v___x_220_);
lean_dec(v___x_212_);
lean_dec(v_a_207_);
lean_dec(v_rsName_202_);
v_a_256_ = lean_ctor_get(v___x_225_, 0);
v_isSharedCheck_263_ = !lean_is_exclusive(v___x_225_);
if (v_isSharedCheck_263_ == 0)
{
v___x_258_ = v___x_225_;
v_isShared_259_ = v_isSharedCheck_263_;
goto v_resetjp_257_;
}
else
{
lean_inc(v_a_256_);
lean_dec(v___x_225_);
v___x_258_ = lean_box(0);
v_isShared_259_ = v_isSharedCheck_263_;
goto v_resetjp_257_;
}
v_resetjp_257_:
{
lean_object* v___x_261_; 
if (v_isShared_259_ == 0)
{
v___x_261_ = v___x_258_;
goto v_reusejp_260_;
}
else
{
lean_object* v_reuseFailAlloc_262_; 
v_reuseFailAlloc_262_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_262_, 0, v_a_256_);
v___x_261_ = v_reuseFailAlloc_262_;
goto v_reusejp_260_;
}
v_reusejp_260_:
{
return v___x_261_;
}
}
}
}
else
{
lean_object* v_a_264_; lean_object* v___x_266_; uint8_t v_isShared_267_; uint8_t v_isSharedCheck_271_; 
lean_dec(v___x_212_);
lean_dec_ref(v___x_211_);
lean_dec_ref(v___x_210_);
lean_dec(v_a_207_);
lean_dec(v_rsName_202_);
v_a_264_ = lean_ctor_get(v___x_217_, 0);
v_isSharedCheck_271_ = !lean_is_exclusive(v___x_217_);
if (v_isSharedCheck_271_ == 0)
{
v___x_266_ = v___x_217_;
v_isShared_267_ = v_isSharedCheck_271_;
goto v_resetjp_265_;
}
else
{
lean_inc(v_a_264_);
lean_dec(v___x_217_);
v___x_266_ = lean_box(0);
v_isShared_267_ = v_isSharedCheck_271_;
goto v_resetjp_265_;
}
v_resetjp_265_:
{
lean_object* v___x_269_; 
if (v_isShared_267_ == 0)
{
v___x_269_ = v___x_266_;
goto v_reusejp_268_;
}
else
{
lean_object* v_reuseFailAlloc_270_; 
v_reuseFailAlloc_270_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_270_, 0, v_a_264_);
v___x_269_ = v_reuseFailAlloc_270_;
goto v_reusejp_268_;
}
v_reusejp_268_:
{
return v___x_269_;
}
}
}
}
else
{
lean_object* v_a_272_; lean_object* v___x_274_; uint8_t v_isShared_275_; uint8_t v_isSharedCheck_279_; 
lean_dec(v_rsName_202_);
v_a_272_ = lean_ctor_get(v___x_206_, 0);
v_isSharedCheck_279_ = !lean_is_exclusive(v___x_206_);
if (v_isSharedCheck_279_ == 0)
{
v___x_274_ = v___x_206_;
v_isShared_275_ = v_isSharedCheck_279_;
goto v_resetjp_273_;
}
else
{
lean_inc(v_a_272_);
lean_dec(v___x_206_);
v___x_274_ = lean_box(0);
v_isShared_275_ = v_isSharedCheck_279_;
goto v_resetjp_273_;
}
v_resetjp_273_:
{
lean_object* v___x_277_; 
if (v_isShared_275_ == 0)
{
v___x_277_ = v___x_274_;
goto v_reusejp_276_;
}
else
{
lean_object* v_reuseFailAlloc_278_; 
v_reuseFailAlloc_278_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_278_, 0, v_a_272_);
v___x_277_ = v_reuseFailAlloc_278_;
goto v_reusejp_276_;
}
v_reusejp_276_:
{
return v___x_277_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_declareRuleSetUnchecked___boxed(lean_object* v_rsName_280_, lean_object* v_isDefault_281_, lean_object* v_a_282_){
_start:
{
uint8_t v_isDefault_boxed_283_; lean_object* v_res_284_; 
v_isDefault_boxed_283_ = lean_unbox(v_isDefault_281_);
v_res_284_ = lp_aesop_Aesop_Frontend_declareRuleSetUnchecked(v_rsName_280_, v_isDefault_boxed_283_);
return v_res_284_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_Frontend_declareRuleSetUnchecked_spec__0(lean_object* v_00_u03b2_285_, lean_object* v_m_286_, lean_object* v_a_287_, lean_object* v_b_288_){
_start:
{
lean_object* v___x_289_; 
v___x_289_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_Frontend_declareRuleSetUnchecked_spec__0___redArg(v_m_286_, v_a_287_, v_b_288_);
return v___x_289_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Aesop_Frontend_declareRuleSetUnchecked_spec__1(lean_object* v_00_u03b2_290_, lean_object* v_m_291_, lean_object* v_a_292_, lean_object* v_b_293_){
_start:
{
lean_object* v___x_294_; 
v___x_294_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Aesop_Frontend_declareRuleSetUnchecked_spec__1___redArg(v_m_291_, v_a_292_, v_b_293_);
return v___x_294_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_Frontend_declareRuleSetUnchecked_spec__0_spec__0(lean_object* v_00_u03b2_295_, lean_object* v_a_296_, lean_object* v_x_297_){
_start:
{
uint8_t v___x_298_; 
v___x_298_ = lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_Frontend_declareRuleSetUnchecked_spec__0_spec__0___redArg(v_a_296_, v_x_297_);
return v___x_298_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_Frontend_declareRuleSetUnchecked_spec__0_spec__0___boxed(lean_object* v_00_u03b2_299_, lean_object* v_a_300_, lean_object* v_x_301_){
_start:
{
uint8_t v_res_302_; lean_object* v_r_303_; 
v_res_302_ = lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_Frontend_declareRuleSetUnchecked_spec__0_spec__0(v_00_u03b2_299_, v_a_300_, v_x_301_);
lean_dec(v_x_301_);
lean_dec(v_a_300_);
v_r_303_ = lean_box(v_res_302_);
return v_r_303_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_Frontend_declareRuleSetUnchecked_spec__0_spec__1(lean_object* v_00_u03b2_304_, lean_object* v_data_305_){
_start:
{
lean_object* v___x_306_; 
v___x_306_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_Frontend_declareRuleSetUnchecked_spec__0_spec__1___redArg(v_data_305_);
return v___x_306_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_Frontend_declareRuleSetUnchecked_spec__0_spec__2(lean_object* v_00_u03b2_307_, lean_object* v_a_308_, lean_object* v_b_309_, lean_object* v_x_310_){
_start:
{
lean_object* v___x_311_; 
v___x_311_ = lp_aesop_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_Frontend_declareRuleSetUnchecked_spec__0_spec__2___redArg(v_a_308_, v_b_309_, v_x_310_);
return v___x_311_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_Frontend_declareRuleSetUnchecked_spec__0_spec__1_spec__2(lean_object* v_00_u03b2_312_, lean_object* v_i_313_, lean_object* v_source_314_, lean_object* v_target_315_){
_start:
{
lean_object* v___x_316_; 
v___x_316_ = lp_aesop___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_Frontend_declareRuleSetUnchecked_spec__0_spec__1_spec__2___redArg(v_i_313_, v_source_314_, v_target_315_);
return v___x_316_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_Frontend_declareRuleSetUnchecked_spec__0_spec__1_spec__2_spec__4(lean_object* v_00_u03b2_317_, lean_object* v_x_318_, lean_object* v_x_319_){
_start:
{
lean_object* v___x_320_; 
v___x_320_ = lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_Frontend_declareRuleSetUnchecked_spec__0_spec__1_spec__2_spec__4___redArg(v_x_318_, v_x_319_);
return v___x_320_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Std_DHashMap_Internal_Raw_u2080_contains___at___00Aesop_Frontend_isRuleSetDeclared_spec__0___redArg(lean_object* v_m_321_, lean_object* v_a_322_){
_start:
{
lean_object* v_buckets_323_; lean_object* v___x_324_; uint64_t v___y_326_; 
v_buckets_323_ = lean_ctor_get(v_m_321_, 1);
v___x_324_ = lean_array_get_size(v_buckets_323_);
if (lean_obj_tag(v_a_322_) == 0)
{
uint64_t v___x_340_; 
v___x_340_ = 1723ULL;
v___y_326_ = v___x_340_;
goto v___jp_325_;
}
else
{
uint64_t v_hash_341_; 
v_hash_341_ = lean_ctor_get_uint64(v_a_322_, sizeof(void*)*2);
v___y_326_ = v_hash_341_;
goto v___jp_325_;
}
v___jp_325_:
{
uint64_t v___x_327_; uint64_t v___x_328_; uint64_t v_fold_329_; uint64_t v___x_330_; uint64_t v___x_331_; uint64_t v___x_332_; size_t v___x_333_; size_t v___x_334_; size_t v___x_335_; size_t v___x_336_; size_t v___x_337_; lean_object* v___x_338_; uint8_t v___x_339_; 
v___x_327_ = 32ULL;
v___x_328_ = lean_uint64_shift_right(v___y_326_, v___x_327_);
v_fold_329_ = lean_uint64_xor(v___y_326_, v___x_328_);
v___x_330_ = 16ULL;
v___x_331_ = lean_uint64_shift_right(v_fold_329_, v___x_330_);
v___x_332_ = lean_uint64_xor(v_fold_329_, v___x_331_);
v___x_333_ = lean_uint64_to_usize(v___x_332_);
v___x_334_ = lean_usize_of_nat(v___x_324_);
v___x_335_ = ((size_t)1ULL);
v___x_336_ = lean_usize_sub(v___x_334_, v___x_335_);
v___x_337_ = lean_usize_land(v___x_333_, v___x_336_);
v___x_338_ = lean_array_uget_borrowed(v_buckets_323_, v___x_337_);
v___x_339_ = lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_Frontend_declareRuleSetUnchecked_spec__0_spec__0___redArg(v_a_322_, v___x_338_);
return v___x_339_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_contains___at___00Aesop_Frontend_isRuleSetDeclared_spec__0___redArg___boxed(lean_object* v_m_342_, lean_object* v_a_343_){
_start:
{
uint8_t v_res_344_; lean_object* v_r_345_; 
v_res_344_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_contains___at___00Aesop_Frontend_isRuleSetDeclared_spec__0___redArg(v_m_342_, v_a_343_);
lean_dec(v_a_343_);
lean_dec_ref(v_m_342_);
v_r_345_ = lean_box(v_res_344_);
return v_r_345_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_isRuleSetDeclared(lean_object* v_rsName_346_){
_start:
{
lean_object* v___x_348_; 
v___x_348_ = lp_aesop_Aesop_getDeclaredRuleSets();
if (lean_obj_tag(v___x_348_) == 0)
{
lean_object* v_a_349_; lean_object* v___x_351_; uint8_t v_isShared_352_; uint8_t v_isSharedCheck_358_; 
v_a_349_ = lean_ctor_get(v___x_348_, 0);
v_isSharedCheck_358_ = !lean_is_exclusive(v___x_348_);
if (v_isSharedCheck_358_ == 0)
{
v___x_351_ = v___x_348_;
v_isShared_352_ = v_isSharedCheck_358_;
goto v_resetjp_350_;
}
else
{
lean_inc(v_a_349_);
lean_dec(v___x_348_);
v___x_351_ = lean_box(0);
v_isShared_352_ = v_isSharedCheck_358_;
goto v_resetjp_350_;
}
v_resetjp_350_:
{
uint8_t v___x_353_; lean_object* v___x_354_; lean_object* v___x_356_; 
v___x_353_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_contains___at___00Aesop_Frontend_isRuleSetDeclared_spec__0___redArg(v_a_349_, v_rsName_346_);
lean_dec(v_a_349_);
v___x_354_ = lean_box(v___x_353_);
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
else
{
lean_object* v_a_359_; lean_object* v___x_361_; uint8_t v_isShared_362_; uint8_t v_isSharedCheck_366_; 
v_a_359_ = lean_ctor_get(v___x_348_, 0);
v_isSharedCheck_366_ = !lean_is_exclusive(v___x_348_);
if (v_isSharedCheck_366_ == 0)
{
v___x_361_ = v___x_348_;
v_isShared_362_ = v_isSharedCheck_366_;
goto v_resetjp_360_;
}
else
{
lean_inc(v_a_359_);
lean_dec(v___x_348_);
v___x_361_ = lean_box(0);
v_isShared_362_ = v_isSharedCheck_366_;
goto v_resetjp_360_;
}
v_resetjp_360_:
{
lean_object* v___x_364_; 
if (v_isShared_362_ == 0)
{
v___x_364_ = v___x_361_;
goto v_reusejp_363_;
}
else
{
lean_object* v_reuseFailAlloc_365_; 
v_reuseFailAlloc_365_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_365_, 0, v_a_359_);
v___x_364_ = v_reuseFailAlloc_365_;
goto v_reusejp_363_;
}
v_reusejp_363_:
{
return v___x_364_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_isRuleSetDeclared___boxed(lean_object* v_rsName_367_, lean_object* v_a_368_){
_start:
{
lean_object* v_res_369_; 
v_res_369_ = lp_aesop_Aesop_Frontend_isRuleSetDeclared(v_rsName_367_);
lean_dec(v_rsName_367_);
return v_res_369_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Std_DHashMap_Internal_Raw_u2080_contains___at___00Aesop_Frontend_isRuleSetDeclared_spec__0(lean_object* v_00_u03b2_370_, lean_object* v_m_371_, lean_object* v_a_372_){
_start:
{
uint8_t v___x_373_; 
v___x_373_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_contains___at___00Aesop_Frontend_isRuleSetDeclared_spec__0___redArg(v_m_371_, v_a_372_);
return v___x_373_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_contains___at___00Aesop_Frontend_isRuleSetDeclared_spec__0___boxed(lean_object* v_00_u03b2_374_, lean_object* v_m_375_, lean_object* v_a_376_){
_start:
{
uint8_t v_res_377_; lean_object* v_r_378_; 
v_res_377_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_contains___at___00Aesop_Frontend_isRuleSetDeclared_spec__0(v_00_u03b2_374_, v_m_375_, v_a_376_);
lean_dec(v_a_376_);
lean_dec_ref(v_m_375_);
v_r_378_ = lean_box(v_res_377_);
return v_r_378_;
}
}
static lean_object* _init_lp_aesop_Aesop_Frontend_checkRuleSetNotDeclared___redArg___lam__0___closed__1(void){
_start:
{
lean_object* v___x_380_; lean_object* v___x_381_; 
v___x_380_ = ((lean_object*)(lp_aesop_Aesop_Frontend_checkRuleSetNotDeclared___redArg___lam__0___closed__0));
v___x_381_ = l_Lean_stringToMessageData(v___x_380_);
return v___x_381_;
}
}
static lean_object* _init_lp_aesop_Aesop_Frontend_checkRuleSetNotDeclared___redArg___lam__0___closed__3(void){
_start:
{
lean_object* v___x_383_; lean_object* v___x_384_; 
v___x_383_ = ((lean_object*)(lp_aesop_Aesop_Frontend_checkRuleSetNotDeclared___redArg___lam__0___closed__2));
v___x_384_ = l_Lean_stringToMessageData(v___x_383_);
return v___x_384_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_checkRuleSetNotDeclared___redArg___lam__0(lean_object* v_toApplicative_385_, lean_object* v_rsName_386_, lean_object* v_inst_387_, lean_object* v_inst_388_, uint8_t v_____do__lift_389_){
_start:
{
if (v_____do__lift_389_ == 0)
{
lean_object* v_toPure_390_; lean_object* v___x_391_; lean_object* v___x_392_; 
lean_dec_ref(v_inst_388_);
lean_dec_ref(v_inst_387_);
lean_dec(v_rsName_386_);
v_toPure_390_ = lean_ctor_get(v_toApplicative_385_, 1);
lean_inc(v_toPure_390_);
lean_dec_ref(v_toApplicative_385_);
v___x_391_ = lean_box(0);
v___x_392_ = lean_apply_2(v_toPure_390_, lean_box(0), v___x_391_);
return v___x_392_;
}
else
{
lean_object* v___x_393_; lean_object* v___x_394_; lean_object* v___x_395_; lean_object* v___x_396_; lean_object* v___x_397_; lean_object* v___x_398_; 
lean_dec_ref(v_toApplicative_385_);
v___x_393_ = lean_obj_once(&lp_aesop_Aesop_Frontend_checkRuleSetNotDeclared___redArg___lam__0___closed__1, &lp_aesop_Aesop_Frontend_checkRuleSetNotDeclared___redArg___lam__0___closed__1_once, _init_lp_aesop_Aesop_Frontend_checkRuleSetNotDeclared___redArg___lam__0___closed__1);
v___x_394_ = l_Lean_MessageData_ofName(v_rsName_386_);
v___x_395_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_395_, 0, v___x_393_);
lean_ctor_set(v___x_395_, 1, v___x_394_);
v___x_396_ = lean_obj_once(&lp_aesop_Aesop_Frontend_checkRuleSetNotDeclared___redArg___lam__0___closed__3, &lp_aesop_Aesop_Frontend_checkRuleSetNotDeclared___redArg___lam__0___closed__3_once, _init_lp_aesop_Aesop_Frontend_checkRuleSetNotDeclared___redArg___lam__0___closed__3);
v___x_397_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_397_, 0, v___x_395_);
lean_ctor_set(v___x_397_, 1, v___x_396_);
v___x_398_ = l_Lean_throwError___redArg(v_inst_387_, v_inst_388_, v___x_397_);
return v___x_398_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_checkRuleSetNotDeclared___redArg___lam__0___boxed(lean_object* v_toApplicative_399_, lean_object* v_rsName_400_, lean_object* v_inst_401_, lean_object* v_inst_402_, lean_object* v_____do__lift_403_){
_start:
{
uint8_t v_____do__lift_131__boxed_404_; lean_object* v_res_405_; 
v_____do__lift_131__boxed_404_ = lean_unbox(v_____do__lift_403_);
v_res_405_ = lp_aesop_Aesop_Frontend_checkRuleSetNotDeclared___redArg___lam__0(v_toApplicative_399_, v_rsName_400_, v_inst_401_, v_inst_402_, v_____do__lift_131__boxed_404_);
return v_res_405_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_checkRuleSetNotDeclared___redArg(lean_object* v_inst_406_, lean_object* v_inst_407_, lean_object* v_inst_408_, lean_object* v_rsName_409_){
_start:
{
lean_object* v_toApplicative_410_; lean_object* v_toBind_411_; lean_object* v___f_412_; lean_object* v___x_413_; lean_object* v___x_414_; lean_object* v___x_415_; 
v_toApplicative_410_ = lean_ctor_get(v_inst_406_, 0);
lean_inc_ref(v_toApplicative_410_);
v_toBind_411_ = lean_ctor_get(v_inst_406_, 1);
lean_inc(v_toBind_411_);
lean_inc(v_rsName_409_);
v___f_412_ = lean_alloc_closure((void*)(lp_aesop_Aesop_Frontend_checkRuleSetNotDeclared___redArg___lam__0___boxed), 5, 4);
lean_closure_set(v___f_412_, 0, v_toApplicative_410_);
lean_closure_set(v___f_412_, 1, v_rsName_409_);
lean_closure_set(v___f_412_, 2, v_inst_406_);
lean_closure_set(v___f_412_, 3, v_inst_407_);
v___x_413_ = lean_alloc_closure((void*)(lp_aesop_Aesop_Frontend_isRuleSetDeclared___boxed), 2, 1);
lean_closure_set(v___x_413_, 0, v_rsName_409_);
v___x_414_ = lean_apply_2(v_inst_408_, lean_box(0), v___x_413_);
v___x_415_ = lean_apply_4(v_toBind_411_, lean_box(0), lean_box(0), v___x_414_, v___f_412_);
return v___x_415_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_checkRuleSetNotDeclared(lean_object* v_m_416_, lean_object* v_inst_417_, lean_object* v_inst_418_, lean_object* v_inst_419_, lean_object* v_rsName_420_){
_start:
{
lean_object* v___x_421_; 
v___x_421_ = lp_aesop_Aesop_Frontend_checkRuleSetNotDeclared___redArg(v_inst_417_, v_inst_418_, v_inst_419_, v_rsName_420_);
return v___x_421_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_declareRuleSet___redArg___lam__0(lean_object* v_rsName_422_, uint8_t v_isDefault_423_, lean_object* v_inst_424_, lean_object* v_____r_425_){
_start:
{
lean_object* v___x_426_; lean_object* v___x_427_; lean_object* v___x_428_; 
v___x_426_ = lean_box(v_isDefault_423_);
v___x_427_ = lean_alloc_closure((void*)(lp_aesop_Aesop_Frontend_declareRuleSetUnchecked___boxed), 3, 2);
lean_closure_set(v___x_427_, 0, v_rsName_422_);
lean_closure_set(v___x_427_, 1, v___x_426_);
v___x_428_ = lean_apply_2(v_inst_424_, lean_box(0), v___x_427_);
return v___x_428_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_declareRuleSet___redArg___lam__0___boxed(lean_object* v_rsName_429_, lean_object* v_isDefault_430_, lean_object* v_inst_431_, lean_object* v_____r_432_){
_start:
{
uint8_t v_isDefault_boxed_433_; lean_object* v_res_434_; 
v_isDefault_boxed_433_ = lean_unbox(v_isDefault_430_);
v_res_434_ = lp_aesop_Aesop_Frontend_declareRuleSet___redArg___lam__0(v_rsName_429_, v_isDefault_boxed_433_, v_inst_431_, v_____r_432_);
return v_res_434_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_declareRuleSet___redArg(lean_object* v_inst_435_, lean_object* v_inst_436_, lean_object* v_inst_437_, lean_object* v_rsName_438_, uint8_t v_isDefault_439_){
_start:
{
lean_object* v_toBind_440_; lean_object* v___x_441_; lean_object* v___f_442_; lean_object* v___x_443_; lean_object* v___x_444_; 
v_toBind_440_ = lean_ctor_get(v_inst_435_, 1);
lean_inc(v_toBind_440_);
v___x_441_ = lean_box(v_isDefault_439_);
lean_inc(v_inst_437_);
lean_inc(v_rsName_438_);
v___f_442_ = lean_alloc_closure((void*)(lp_aesop_Aesop_Frontend_declareRuleSet___redArg___lam__0___boxed), 4, 3);
lean_closure_set(v___f_442_, 0, v_rsName_438_);
lean_closure_set(v___f_442_, 1, v___x_441_);
lean_closure_set(v___f_442_, 2, v_inst_437_);
v___x_443_ = lp_aesop_Aesop_Frontend_checkRuleSetNotDeclared___redArg(v_inst_435_, v_inst_436_, v_inst_437_, v_rsName_438_);
v___x_444_ = lean_apply_4(v_toBind_440_, lean_box(0), lean_box(0), v___x_443_, v___f_442_);
return v___x_444_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_declareRuleSet___redArg___boxed(lean_object* v_inst_445_, lean_object* v_inst_446_, lean_object* v_inst_447_, lean_object* v_rsName_448_, lean_object* v_isDefault_449_){
_start:
{
uint8_t v_isDefault_boxed_450_; lean_object* v_res_451_; 
v_isDefault_boxed_450_ = lean_unbox(v_isDefault_449_);
v_res_451_ = lp_aesop_Aesop_Frontend_declareRuleSet___redArg(v_inst_445_, v_inst_446_, v_inst_447_, v_rsName_448_, v_isDefault_boxed_450_);
return v_res_451_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_declareRuleSet(lean_object* v_m_452_, lean_object* v_inst_453_, lean_object* v_inst_454_, lean_object* v_inst_455_, lean_object* v_rsName_456_, uint8_t v_isDefault_457_){
_start:
{
lean_object* v___x_458_; 
v___x_458_ = lp_aesop_Aesop_Frontend_declareRuleSet___redArg(v_inst_453_, v_inst_454_, v_inst_455_, v_rsName_456_, v_isDefault_457_);
return v___x_458_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_declareRuleSet___boxed(lean_object* v_m_459_, lean_object* v_inst_460_, lean_object* v_inst_461_, lean_object* v_inst_462_, lean_object* v_rsName_463_, lean_object* v_isDefault_464_){
_start:
{
uint8_t v_isDefault_boxed_465_; lean_object* v_res_466_; 
v_isDefault_boxed_465_ = lean_unbox(v_isDefault_464_);
v_res_466_ = lp_aesop_Aesop_Frontend_declareRuleSet(v_m_459_, v_inst_460_, v_inst_461_, v_inst_462_, v_rsName_463_, v_isDefault_boxed_465_);
return v_res_466_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Aesop_Frontend_Extension_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Extension_3818590508____hygCtx___hyg_2__spec__0(lean_object* v_as_467_, size_t v_i_468_, size_t v_stop_469_, lean_object* v_b_470_){
_start:
{
uint8_t v___x_472_; 
v___x_472_ = lean_usize_dec_eq(v_i_468_, v_stop_469_);
if (v___x_472_ == 0)
{
lean_object* v___x_473_; uint8_t v___x_474_; lean_object* v___x_475_; 
v___x_473_ = lean_array_uget_borrowed(v_as_467_, v_i_468_);
v___x_474_ = 1;
lean_inc(v___x_473_);
v___x_475_ = lp_aesop_Aesop_Frontend_declareRuleSetUnchecked(v___x_473_, v___x_474_);
if (lean_obj_tag(v___x_475_) == 0)
{
lean_object* v_a_476_; size_t v___x_477_; size_t v___x_478_; 
v_a_476_ = lean_ctor_get(v___x_475_, 0);
lean_inc(v_a_476_);
lean_dec_ref_known(v___x_475_, 1);
v___x_477_ = ((size_t)1ULL);
v___x_478_ = lean_usize_add(v_i_468_, v___x_477_);
v_i_468_ = v___x_478_;
v_b_470_ = v_a_476_;
goto _start;
}
else
{
return v___x_475_;
}
}
else
{
lean_object* v___x_480_; 
v___x_480_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_480_, 0, v_b_470_);
return v___x_480_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Aesop_Frontend_Extension_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Extension_3818590508____hygCtx___hyg_2__spec__0___boxed(lean_object* v_as_481_, lean_object* v_i_482_, lean_object* v_stop_483_, lean_object* v_b_484_, lean_object* v___y_485_){
_start:
{
size_t v_i_boxed_486_; size_t v_stop_boxed_487_; lean_object* v_res_488_; 
v_i_boxed_486_ = lean_unbox_usize(v_i_482_);
lean_dec(v_i_482_);
v_stop_boxed_487_ = lean_unbox_usize(v_stop_483_);
lean_dec(v_stop_483_);
v_res_488_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Aesop_Frontend_Extension_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Extension_3818590508____hygCtx___hyg_2__spec__0(v_as_481_, v_i_boxed_486_, v_stop_boxed_487_, v_b_484_);
lean_dec_ref(v_as_481_);
return v_res_488_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Frontend_Extension_0__Aesop_Frontend_initFn___closed__0_00___x40_Aesop_Frontend_Extension_3818590508____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_489_; lean_object* v___x_490_; 
v___x_489_ = lp_aesop_Aesop_builtinRuleSetNames;
v___x_490_ = lean_array_get_size(v___x_489_);
return v___x_490_;
}
}
static uint8_t _init_lp_aesop___private_Aesop_Frontend_Extension_0__Aesop_Frontend_initFn___closed__1_00___x40_Aesop_Frontend_Extension_3818590508____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_491_; lean_object* v___x_492_; uint8_t v___x_493_; 
v___x_491_ = lean_obj_once(&lp_aesop___private_Aesop_Frontend_Extension_0__Aesop_Frontend_initFn___closed__0_00___x40_Aesop_Frontend_Extension_3818590508____hygCtx___hyg_2_, &lp_aesop___private_Aesop_Frontend_Extension_0__Aesop_Frontend_initFn___closed__0_00___x40_Aesop_Frontend_Extension_3818590508____hygCtx___hyg_2__once, _init_lp_aesop___private_Aesop_Frontend_Extension_0__Aesop_Frontend_initFn___closed__0_00___x40_Aesop_Frontend_Extension_3818590508____hygCtx___hyg_2_);
v___x_492_ = lean_unsigned_to_nat(0u);
v___x_493_ = lean_nat_dec_lt(v___x_492_, v___x_491_);
return v___x_493_;
}
}
static uint8_t _init_lp_aesop___private_Aesop_Frontend_Extension_0__Aesop_Frontend_initFn___closed__2_00___x40_Aesop_Frontend_Extension_3818590508____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_494_; uint8_t v___x_495_; 
v___x_494_ = lean_obj_once(&lp_aesop___private_Aesop_Frontend_Extension_0__Aesop_Frontend_initFn___closed__0_00___x40_Aesop_Frontend_Extension_3818590508____hygCtx___hyg_2_, &lp_aesop___private_Aesop_Frontend_Extension_0__Aesop_Frontend_initFn___closed__0_00___x40_Aesop_Frontend_Extension_3818590508____hygCtx___hyg_2__once, _init_lp_aesop___private_Aesop_Frontend_Extension_0__Aesop_Frontend_initFn___closed__0_00___x40_Aesop_Frontend_Extension_3818590508____hygCtx___hyg_2_);
v___x_495_ = lean_nat_dec_le(v___x_494_, v___x_494_);
return v___x_495_;
}
}
static size_t _init_lp_aesop___private_Aesop_Frontend_Extension_0__Aesop_Frontend_initFn___closed__3_00___x40_Aesop_Frontend_Extension_3818590508____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_496_; size_t v___x_497_; 
v___x_496_ = lean_obj_once(&lp_aesop___private_Aesop_Frontend_Extension_0__Aesop_Frontend_initFn___closed__0_00___x40_Aesop_Frontend_Extension_3818590508____hygCtx___hyg_2_, &lp_aesop___private_Aesop_Frontend_Extension_0__Aesop_Frontend_initFn___closed__0_00___x40_Aesop_Frontend_Extension_3818590508____hygCtx___hyg_2__once, _init_lp_aesop___private_Aesop_Frontend_Extension_0__Aesop_Frontend_initFn___closed__0_00___x40_Aesop_Frontend_Extension_3818590508____hygCtx___hyg_2_);
v___x_497_ = lean_usize_of_nat(v___x_496_);
return v___x_497_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Frontend_Extension_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Extension_3818590508____hygCtx___hyg_2_(){
_start:
{
lean_object* v___x_499_; lean_object* v___x_500_; uint8_t v___x_501_; 
v___x_499_ = lp_aesop_Aesop_builtinRuleSetNames;
v___x_500_ = lean_box(0);
v___x_501_ = lean_uint8_once(&lp_aesop___private_Aesop_Frontend_Extension_0__Aesop_Frontend_initFn___closed__1_00___x40_Aesop_Frontend_Extension_3818590508____hygCtx___hyg_2_, &lp_aesop___private_Aesop_Frontend_Extension_0__Aesop_Frontend_initFn___closed__1_00___x40_Aesop_Frontend_Extension_3818590508____hygCtx___hyg_2__once, _init_lp_aesop___private_Aesop_Frontend_Extension_0__Aesop_Frontend_initFn___closed__1_00___x40_Aesop_Frontend_Extension_3818590508____hygCtx___hyg_2_);
if (v___x_501_ == 0)
{
lean_object* v___x_502_; 
v___x_502_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_502_, 0, v___x_500_);
return v___x_502_;
}
else
{
uint8_t v___x_503_; 
v___x_503_ = lean_uint8_once(&lp_aesop___private_Aesop_Frontend_Extension_0__Aesop_Frontend_initFn___closed__2_00___x40_Aesop_Frontend_Extension_3818590508____hygCtx___hyg_2_, &lp_aesop___private_Aesop_Frontend_Extension_0__Aesop_Frontend_initFn___closed__2_00___x40_Aesop_Frontend_Extension_3818590508____hygCtx___hyg_2__once, _init_lp_aesop___private_Aesop_Frontend_Extension_0__Aesop_Frontend_initFn___closed__2_00___x40_Aesop_Frontend_Extension_3818590508____hygCtx___hyg_2_);
if (v___x_503_ == 0)
{
if (v___x_501_ == 0)
{
lean_object* v___x_504_; 
v___x_504_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_504_, 0, v___x_500_);
return v___x_504_;
}
else
{
size_t v___x_505_; size_t v___x_506_; lean_object* v___x_507_; 
v___x_505_ = ((size_t)0ULL);
v___x_506_ = lean_usize_once(&lp_aesop___private_Aesop_Frontend_Extension_0__Aesop_Frontend_initFn___closed__3_00___x40_Aesop_Frontend_Extension_3818590508____hygCtx___hyg_2_, &lp_aesop___private_Aesop_Frontend_Extension_0__Aesop_Frontend_initFn___closed__3_00___x40_Aesop_Frontend_Extension_3818590508____hygCtx___hyg_2__once, _init_lp_aesop___private_Aesop_Frontend_Extension_0__Aesop_Frontend_initFn___closed__3_00___x40_Aesop_Frontend_Extension_3818590508____hygCtx___hyg_2_);
v___x_507_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Aesop_Frontend_Extension_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Extension_3818590508____hygCtx___hyg_2__spec__0(v___x_499_, v___x_505_, v___x_506_, v___x_500_);
return v___x_507_;
}
}
else
{
size_t v___x_508_; size_t v___x_509_; lean_object* v___x_510_; 
v___x_508_ = ((size_t)0ULL);
v___x_509_ = lean_usize_once(&lp_aesop___private_Aesop_Frontend_Extension_0__Aesop_Frontend_initFn___closed__3_00___x40_Aesop_Frontend_Extension_3818590508____hygCtx___hyg_2_, &lp_aesop___private_Aesop_Frontend_Extension_0__Aesop_Frontend_initFn___closed__3_00___x40_Aesop_Frontend_Extension_3818590508____hygCtx___hyg_2__once, _init_lp_aesop___private_Aesop_Frontend_Extension_0__Aesop_Frontend_initFn___closed__3_00___x40_Aesop_Frontend_Extension_3818590508____hygCtx___hyg_2_);
v___x_510_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Aesop_Frontend_Extension_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Extension_3818590508____hygCtx___hyg_2__spec__0(v___x_499_, v___x_508_, v___x_509_, v___x_500_);
return v___x_510_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Frontend_Extension_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Extension_3818590508____hygCtx___hyg_2____boxed(lean_object* v_a_511_){
_start:
{
lean_object* v_res_512_; 
v_res_512_ = lp_aesop___private_Aesop_Frontend_Extension_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Extension_3818590508____hygCtx___hyg_2_();
return v_res_512_;
}
}
static lean_object* _init_lp_aesop_Aesop_Frontend_getGlobalRuleSetData___redArg___lam__0___closed__1(void){
_start:
{
lean_object* v___x_514_; lean_object* v___x_515_; 
v___x_514_ = ((lean_object*)(lp_aesop_Aesop_Frontend_getGlobalRuleSetData___redArg___lam__0___closed__0));
v___x_515_ = l_Lean_stringToMessageData(v___x_514_);
return v___x_515_;
}
}
static lean_object* _init_lp_aesop_Aesop_Frontend_getGlobalRuleSetData___redArg___lam__0___closed__3(void){
_start:
{
lean_object* v___x_517_; lean_object* v___x_518_; 
v___x_517_ = ((lean_object*)(lp_aesop_Aesop_Frontend_getGlobalRuleSetData___redArg___lam__0___closed__2));
v___x_518_ = l_Lean_stringToMessageData(v___x_517_);
return v___x_518_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_getGlobalRuleSetData___redArg___lam__0(lean_object* v_snd_519_, lean_object* v_val_520_, lean_object* v_fst_521_, lean_object* v_fst_522_, lean_object* v_toPure_523_, lean_object* v_inst_524_, lean_object* v_inst_525_, lean_object* v_____x_526_){
_start:
{
if (lean_obj_tag(v_____x_526_) == 1)
{
lean_object* v_val_527_; lean_object* v___x_528_; lean_object* v___x_529_; lean_object* v___x_530_; lean_object* v___x_531_; lean_object* v___x_532_; 
lean_dec_ref(v_inst_525_);
lean_dec_ref(v_inst_524_);
v_val_527_ = lean_ctor_get(v_____x_526_, 0);
lean_inc(v_val_527_);
v___x_528_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_528_, 0, v_snd_519_);
lean_ctor_set(v___x_528_, 1, v_val_527_);
v___x_529_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_529_, 0, v_val_520_);
lean_ctor_set(v___x_529_, 1, v___x_528_);
v___x_530_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_530_, 0, v_fst_521_);
lean_ctor_set(v___x_530_, 1, v___x_529_);
v___x_531_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_531_, 0, v_fst_522_);
lean_ctor_set(v___x_531_, 1, v___x_530_);
v___x_532_ = lean_apply_2(v_toPure_523_, lean_box(0), v___x_531_);
return v___x_532_;
}
else
{
lean_object* v___x_533_; lean_object* v___x_534_; lean_object* v___x_535_; lean_object* v___x_536_; lean_object* v___x_537_; lean_object* v___x_538_; 
lean_dec(v_toPure_523_);
lean_dec_ref(v_fst_522_);
lean_dec_ref(v_val_520_);
lean_dec(v_snd_519_);
v___x_533_ = lean_obj_once(&lp_aesop_Aesop_Frontend_getGlobalRuleSetData___redArg___lam__0___closed__1, &lp_aesop_Aesop_Frontend_getGlobalRuleSetData___redArg___lam__0___closed__1_once, _init_lp_aesop_Aesop_Frontend_getGlobalRuleSetData___redArg___lam__0___closed__1);
v___x_534_ = l_Lean_MessageData_ofName(v_fst_521_);
v___x_535_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_535_, 0, v___x_533_);
lean_ctor_set(v___x_535_, 1, v___x_534_);
v___x_536_ = lean_obj_once(&lp_aesop_Aesop_Frontend_getGlobalRuleSetData___redArg___lam__0___closed__3, &lp_aesop_Aesop_Frontend_getGlobalRuleSetData___redArg___lam__0___closed__3_once, _init_lp_aesop_Aesop_Frontend_getGlobalRuleSetData___redArg___lam__0___closed__3);
v___x_537_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_537_, 0, v___x_535_);
lean_ctor_set(v___x_537_, 1, v___x_536_);
v___x_538_ = l_Lean_throwError___redArg(v_inst_524_, v_inst_525_, v___x_537_);
return v___x_538_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_getGlobalRuleSetData___redArg___lam__0___boxed(lean_object* v_snd_539_, lean_object* v_val_540_, lean_object* v_fst_541_, lean_object* v_fst_542_, lean_object* v_toPure_543_, lean_object* v_inst_544_, lean_object* v_inst_545_, lean_object* v_____x_546_){
_start:
{
lean_object* v_res_547_; 
v_res_547_ = lp_aesop_Aesop_Frontend_getGlobalRuleSetData___redArg___lam__0(v_snd_539_, v_val_540_, v_fst_541_, v_fst_542_, v_toPure_543_, v_inst_544_, v_inst_545_, v_____x_546_);
lean_dec(v_____x_546_);
return v_res_547_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_getGlobalRuleSetData___redArg___lam__1(lean_object* v_snd_548_, lean_object* v_fst_549_, lean_object* v_fst_550_, lean_object* v_toPure_551_, lean_object* v_inst_552_, lean_object* v_inst_553_, lean_object* v_inst_554_, lean_object* v_toBind_555_, lean_object* v_____x_556_){
_start:
{
if (lean_obj_tag(v_____x_556_) == 1)
{
lean_object* v_val_557_; lean_object* v___f_558_; lean_object* v___x_559_; lean_object* v___x_560_; lean_object* v___x_561_; 
v_val_557_ = lean_ctor_get(v_____x_556_, 0);
lean_inc(v_val_557_);
lean_dec_ref_known(v_____x_556_, 1);
lean_inc(v_fst_549_);
v___f_558_ = lean_alloc_closure((void*)(lp_aesop_Aesop_Frontend_getGlobalRuleSetData___redArg___lam__0___boxed), 8, 7);
lean_closure_set(v___f_558_, 0, v_snd_548_);
lean_closure_set(v___f_558_, 1, v_val_557_);
lean_closure_set(v___f_558_, 2, v_fst_549_);
lean_closure_set(v___f_558_, 3, v_fst_550_);
lean_closure_set(v___f_558_, 4, v_toPure_551_);
lean_closure_set(v___f_558_, 5, v_inst_552_);
lean_closure_set(v___f_558_, 6, v_inst_553_);
v___x_559_ = lean_alloc_closure((void*)(l_Lean_Meta_Simp_getSimprocExtension_x3f___boxed), 2, 1);
lean_closure_set(v___x_559_, 0, v_fst_549_);
v___x_560_ = lean_apply_2(v_inst_554_, lean_box(0), v___x_559_);
v___x_561_ = lean_apply_4(v_toBind_555_, lean_box(0), lean_box(0), v___x_560_, v___f_558_);
return v___x_561_;
}
else
{
lean_object* v___x_562_; lean_object* v___x_563_; lean_object* v___x_564_; lean_object* v___x_565_; lean_object* v___x_566_; lean_object* v___x_567_; 
lean_dec(v_____x_556_);
lean_dec(v_toBind_555_);
lean_dec(v_inst_554_);
lean_dec(v_toPure_551_);
lean_dec_ref(v_fst_550_);
lean_dec(v_snd_548_);
v___x_562_ = lean_obj_once(&lp_aesop_Aesop_Frontend_getGlobalRuleSetData___redArg___lam__0___closed__1, &lp_aesop_Aesop_Frontend_getGlobalRuleSetData___redArg___lam__0___closed__1_once, _init_lp_aesop_Aesop_Frontend_getGlobalRuleSetData___redArg___lam__0___closed__1);
v___x_563_ = l_Lean_MessageData_ofName(v_fst_549_);
v___x_564_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_564_, 0, v___x_562_);
lean_ctor_set(v___x_564_, 1, v___x_563_);
v___x_565_ = lean_obj_once(&lp_aesop_Aesop_Frontend_getGlobalRuleSetData___redArg___lam__0___closed__3, &lp_aesop_Aesop_Frontend_getGlobalRuleSetData___redArg___lam__0___closed__3_once, _init_lp_aesop_Aesop_Frontend_getGlobalRuleSetData___redArg___lam__0___closed__3);
v___x_566_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_566_, 0, v___x_564_);
lean_ctor_set(v___x_566_, 1, v___x_565_);
v___x_567_ = l_Lean_throwError___redArg(v_inst_552_, v_inst_553_, v___x_566_);
return v___x_567_;
}
}
}
static lean_object* _init_lp_aesop_Aesop_Frontend_getGlobalRuleSetData___redArg___lam__2___closed__1(void){
_start:
{
lean_object* v___x_569_; lean_object* v___x_570_; 
v___x_569_ = ((lean_object*)(lp_aesop_Aesop_Frontend_getGlobalRuleSetData___redArg___lam__2___closed__0));
v___x_570_ = l_Lean_stringToMessageData(v___x_569_);
return v___x_570_;
}
}
static lean_object* _init_lp_aesop_Aesop_Frontend_getGlobalRuleSetData___redArg___lam__2___closed__3(void){
_start:
{
lean_object* v___x_572_; lean_object* v___x_573_; 
v___x_572_ = ((lean_object*)(lp_aesop_Aesop_Frontend_getGlobalRuleSetData___redArg___lam__2___closed__2));
v___x_573_ = l_Lean_stringToMessageData(v___x_572_);
return v___x_573_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_getGlobalRuleSetData___redArg___lam__2(lean_object* v___x_574_, lean_object* v___x_575_, lean_object* v_rsName_576_, lean_object* v_toPure_577_, lean_object* v_inst_578_, lean_object* v_inst_579_, lean_object* v_inst_580_, lean_object* v_toBind_581_, lean_object* v_inst_582_, lean_object* v_____do__lift_583_){
_start:
{
lean_object* v___x_584_; 
lean_inc(v_rsName_576_);
v___x_584_ = l_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___redArg(v___x_574_, v___x_575_, v_____do__lift_583_, v_rsName_576_);
if (lean_obj_tag(v___x_584_) == 1)
{
lean_object* v_val_585_; lean_object* v_snd_586_; lean_object* v_fst_587_; lean_object* v_fst_588_; lean_object* v_snd_589_; lean_object* v___f_590_; lean_object* v___x_591_; lean_object* v___x_592_; lean_object* v___x_593_; 
lean_dec(v_rsName_576_);
v_val_585_ = lean_ctor_get(v___x_584_, 0);
lean_inc(v_val_585_);
lean_dec_ref_known(v___x_584_, 1);
v_snd_586_ = lean_ctor_get(v_val_585_, 1);
lean_inc(v_snd_586_);
v_fst_587_ = lean_ctor_get(v_val_585_, 0);
lean_inc(v_fst_587_);
lean_dec(v_val_585_);
v_fst_588_ = lean_ctor_get(v_snd_586_, 0);
lean_inc_n(v_fst_588_, 2);
v_snd_589_ = lean_ctor_get(v_snd_586_, 1);
lean_inc(v_snd_589_);
lean_dec(v_snd_586_);
lean_inc(v_toBind_581_);
v___f_590_ = lean_alloc_closure((void*)(lp_aesop_Aesop_Frontend_getGlobalRuleSetData___redArg___lam__1), 9, 8);
lean_closure_set(v___f_590_, 0, v_snd_589_);
lean_closure_set(v___f_590_, 1, v_fst_588_);
lean_closure_set(v___f_590_, 2, v_fst_587_);
lean_closure_set(v___f_590_, 3, v_toPure_577_);
lean_closure_set(v___f_590_, 4, v_inst_578_);
lean_closure_set(v___f_590_, 5, v_inst_579_);
lean_closure_set(v___f_590_, 6, v_inst_580_);
lean_closure_set(v___f_590_, 7, v_toBind_581_);
v___x_591_ = lean_alloc_closure((void*)(l_Lean_Meta_getSimpExtension_x3f___boxed), 4, 1);
lean_closure_set(v___x_591_, 0, v_fst_588_);
v___x_592_ = lean_apply_2(v_inst_582_, lean_box(0), v___x_591_);
v___x_593_ = lean_apply_4(v_toBind_581_, lean_box(0), lean_box(0), v___x_592_, v___f_590_);
return v___x_593_;
}
else
{
lean_object* v___x_594_; lean_object* v___x_595_; lean_object* v___x_596_; lean_object* v___x_597_; lean_object* v___x_598_; lean_object* v___x_599_; 
lean_dec(v___x_584_);
lean_dec(v_inst_582_);
lean_dec(v_toBind_581_);
lean_dec(v_inst_580_);
lean_dec(v_toPure_577_);
v___x_594_ = lean_obj_once(&lp_aesop_Aesop_Frontend_getGlobalRuleSetData___redArg___lam__2___closed__1, &lp_aesop_Aesop_Frontend_getGlobalRuleSetData___redArg___lam__2___closed__1_once, _init_lp_aesop_Aesop_Frontend_getGlobalRuleSetData___redArg___lam__2___closed__1);
v___x_595_ = l_Lean_MessageData_ofName(v_rsName_576_);
v___x_596_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_596_, 0, v___x_594_);
lean_ctor_set(v___x_596_, 1, v___x_595_);
v___x_597_ = lean_obj_once(&lp_aesop_Aesop_Frontend_getGlobalRuleSetData___redArg___lam__2___closed__3, &lp_aesop_Aesop_Frontend_getGlobalRuleSetData___redArg___lam__2___closed__3_once, _init_lp_aesop_Aesop_Frontend_getGlobalRuleSetData___redArg___lam__2___closed__3);
v___x_598_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_598_, 0, v___x_596_);
lean_ctor_set(v___x_598_, 1, v___x_597_);
v___x_599_ = l_Lean_throwError___redArg(v_inst_578_, v_inst_579_, v___x_598_);
return v___x_599_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_getGlobalRuleSetData___redArg___lam__2___boxed(lean_object* v___x_600_, lean_object* v___x_601_, lean_object* v_rsName_602_, lean_object* v_toPure_603_, lean_object* v_inst_604_, lean_object* v_inst_605_, lean_object* v_inst_606_, lean_object* v_toBind_607_, lean_object* v_inst_608_, lean_object* v_____do__lift_609_){
_start:
{
lean_object* v_res_610_; 
v_res_610_ = lp_aesop_Aesop_Frontend_getGlobalRuleSetData___redArg___lam__2(v___x_600_, v___x_601_, v_rsName_602_, v_toPure_603_, v_inst_604_, v_inst_605_, v_inst_606_, v_toBind_607_, v_inst_608_, v_____do__lift_609_);
lean_dec_ref(v_____do__lift_609_);
return v_res_610_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_getGlobalRuleSetData___redArg(lean_object* v_inst_614_, lean_object* v_inst_615_, lean_object* v_inst_616_, lean_object* v_inst_617_, lean_object* v_rsName_618_){
_start:
{
lean_object* v_toApplicative_619_; lean_object* v_toBind_620_; lean_object* v_toPure_621_; lean_object* v___x_622_; lean_object* v___x_623_; lean_object* v___x_624_; lean_object* v___x_625_; lean_object* v___f_626_; lean_object* v___x_627_; 
v_toApplicative_619_ = lean_ctor_get(v_inst_614_, 0);
v_toBind_620_ = lean_ctor_get(v_inst_614_, 1);
lean_inc_n(v_toBind_620_, 2);
v_toPure_621_ = lean_ctor_get(v_toApplicative_619_, 1);
lean_inc(v_toPure_621_);
v___x_622_ = ((lean_object*)(lp_aesop_Aesop_Frontend_getGlobalRuleSetData___redArg___closed__0));
v___x_623_ = ((lean_object*)(lp_aesop_Aesop_Frontend_getGlobalRuleSetData___redArg___closed__1));
v___x_624_ = ((lean_object*)(lp_aesop_Aesop_Frontend_getGlobalRuleSetData___redArg___closed__2));
lean_inc(v_inst_616_);
v___x_625_ = lean_apply_2(v_inst_616_, lean_box(0), v___x_624_);
v___f_626_ = lean_alloc_closure((void*)(lp_aesop_Aesop_Frontend_getGlobalRuleSetData___redArg___lam__2___boxed), 10, 9);
lean_closure_set(v___f_626_, 0, v___x_622_);
lean_closure_set(v___f_626_, 1, v___x_623_);
lean_closure_set(v___f_626_, 2, v_rsName_618_);
lean_closure_set(v___f_626_, 3, v_toPure_621_);
lean_closure_set(v___f_626_, 4, v_inst_614_);
lean_closure_set(v___f_626_, 5, v_inst_615_);
lean_closure_set(v___f_626_, 6, v_inst_616_);
lean_closure_set(v___f_626_, 7, v_toBind_620_);
lean_closure_set(v___f_626_, 8, v_inst_617_);
v___x_627_ = lean_apply_4(v_toBind_620_, lean_box(0), lean_box(0), v___x_625_, v___f_626_);
return v___x_627_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_getGlobalRuleSetData(lean_object* v_m_628_, lean_object* v_inst_629_, lean_object* v_inst_630_, lean_object* v_inst_631_, lean_object* v_inst_632_, lean_object* v_rsName_633_){
_start:
{
lean_object* v___x_634_; 
v___x_634_ = lp_aesop_Aesop_Frontend_getGlobalRuleSetData___redArg(v_inst_629_, v_inst_630_, v_inst_631_, v_inst_632_, v_rsName_633_);
return v___x_634_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_getGlobalRuleSetFromData___redArg___lam__0(lean_object* v_ext_635_, lean_object* v_simpExt_636_, lean_object* v_simprocExt_637_, lean_object* v___x_638_, lean_object* v___x_639_, lean_object* v___x_640_, lean_object* v_toPure_641_, lean_object* v_env_642_){
_start:
{
lean_object* v_ext_643_; lean_object* v_toEnvExtension_644_; lean_object* v_ext_645_; lean_object* v_toEnvExtension_646_; lean_object* v_ext_647_; lean_object* v_toEnvExtension_648_; lean_object* v_asyncMode_649_; lean_object* v_asyncMode_650_; lean_object* v_asyncMode_651_; lean_object* v_base_652_; lean_object* v_simpTheorems_653_; lean_object* v_simprocs_654_; lean_object* v___x_655_; lean_object* v___x_656_; 
v_ext_643_ = lean_ctor_get(v_ext_635_, 1);
v_toEnvExtension_644_ = lean_ctor_get(v_ext_643_, 0);
v_ext_645_ = lean_ctor_get(v_simpExt_636_, 1);
v_toEnvExtension_646_ = lean_ctor_get(v_ext_645_, 0);
v_ext_647_ = lean_ctor_get(v_simprocExt_637_, 1);
v_toEnvExtension_648_ = lean_ctor_get(v_ext_647_, 0);
v_asyncMode_649_ = lean_ctor_get(v_toEnvExtension_644_, 2);
v_asyncMode_650_ = lean_ctor_get(v_toEnvExtension_646_, 2);
v_asyncMode_651_ = lean_ctor_get(v_toEnvExtension_648_, 2);
lean_inc_ref_n(v_env_642_, 2);
v_base_652_ = l_Lean_ScopedEnvExtension_getState___redArg(v___x_638_, v_ext_635_, v_env_642_, v_asyncMode_649_);
v_simpTheorems_653_ = l_Lean_ScopedEnvExtension_getState___redArg(v___x_639_, v_simpExt_636_, v_env_642_, v_asyncMode_650_);
v_simprocs_654_ = l_Lean_ScopedEnvExtension_getState___redArg(v___x_640_, v_simprocExt_637_, v_env_642_, v_asyncMode_651_);
v___x_655_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_655_, 0, v_base_652_);
lean_ctor_set(v___x_655_, 1, v_simpTheorems_653_);
lean_ctor_set(v___x_655_, 2, v_simprocs_654_);
v___x_656_ = lean_apply_2(v_toPure_641_, lean_box(0), v___x_655_);
return v___x_656_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_getGlobalRuleSetFromData___redArg___lam__0___boxed(lean_object* v_ext_657_, lean_object* v_simpExt_658_, lean_object* v_simprocExt_659_, lean_object* v___x_660_, lean_object* v___x_661_, lean_object* v___x_662_, lean_object* v_toPure_663_, lean_object* v_env_664_){
_start:
{
lean_object* v_res_665_; 
v_res_665_ = lp_aesop_Aesop_Frontend_getGlobalRuleSetFromData___redArg___lam__0(v_ext_657_, v_simpExt_658_, v_simprocExt_659_, v___x_660_, v___x_661_, v___x_662_, v_toPure_663_, v_env_664_);
lean_dec_ref(v___x_662_);
lean_dec_ref(v___x_661_);
lean_dec_ref(v___x_660_);
lean_dec_ref(v_simprocExt_659_);
lean_dec_ref(v_simpExt_658_);
lean_dec_ref(v_ext_657_);
return v_res_665_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_getGlobalRuleSetFromData___redArg(lean_object* v_inst_666_, lean_object* v_inst_667_, lean_object* v_ext_668_, lean_object* v_simpExt_669_, lean_object* v_simprocExt_670_){
_start:
{
lean_object* v_toApplicative_671_; lean_object* v_toBind_672_; lean_object* v_getEnv_673_; lean_object* v_toPure_674_; lean_object* v___x_675_; lean_object* v___x_676_; lean_object* v___x_677_; lean_object* v___f_678_; lean_object* v___x_679_; 
v_toApplicative_671_ = lean_ctor_get(v_inst_666_, 0);
lean_inc_ref(v_toApplicative_671_);
v_toBind_672_ = lean_ctor_get(v_inst_666_, 1);
lean_inc(v_toBind_672_);
lean_dec_ref(v_inst_666_);
v_getEnv_673_ = lean_ctor_get(v_inst_667_, 0);
lean_inc(v_getEnv_673_);
lean_dec_ref(v_inst_667_);
v_toPure_674_ = lean_ctor_get(v_toApplicative_671_, 1);
lean_inc(v_toPure_674_);
lean_dec_ref(v_toApplicative_671_);
v___x_675_ = lp_aesop_Aesop_instInhabitedBaseRuleSet_default;
v___x_676_ = l_Lean_Meta_instInhabitedSimpTheorems_default;
v___x_677_ = l_Lean_Meta_Simp_instInhabitedSimprocs_default;
v___f_678_ = lean_alloc_closure((void*)(lp_aesop_Aesop_Frontend_getGlobalRuleSetFromData___redArg___lam__0___boxed), 8, 7);
lean_closure_set(v___f_678_, 0, v_ext_668_);
lean_closure_set(v___f_678_, 1, v_simpExt_669_);
lean_closure_set(v___f_678_, 2, v_simprocExt_670_);
lean_closure_set(v___f_678_, 3, v___x_675_);
lean_closure_set(v___f_678_, 4, v___x_676_);
lean_closure_set(v___f_678_, 5, v___x_677_);
lean_closure_set(v___f_678_, 6, v_toPure_674_);
v___x_679_ = lean_apply_4(v_toBind_672_, lean_box(0), lean_box(0), v_getEnv_673_, v___f_678_);
return v___x_679_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_getGlobalRuleSetFromData(lean_object* v_m_680_, lean_object* v_inst_681_, lean_object* v_inst_682_, lean_object* v_ext_683_, lean_object* v_simpExt_684_, lean_object* v_simprocExt_685_){
_start:
{
lean_object* v___x_686_; 
v___x_686_ = lp_aesop_Aesop_Frontend_getGlobalRuleSetFromData___redArg(v_inst_681_, v_inst_682_, v_ext_683_, v_simpExt_684_, v_simprocExt_685_);
return v___x_686_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_getGlobalRuleSetFromData___at___00Aesop_Frontend_getGlobalRuleSet_spec__1___redArg(lean_object* v_ext_687_, lean_object* v_simpExt_688_, lean_object* v_simprocExt_689_, lean_object* v___y_690_){
_start:
{
lean_object* v___x_692_; lean_object* v_ext_693_; lean_object* v_toEnvExtension_694_; lean_object* v_ext_695_; lean_object* v_toEnvExtension_696_; lean_object* v_ext_697_; lean_object* v_toEnvExtension_698_; lean_object* v_env_699_; lean_object* v_asyncMode_700_; lean_object* v_asyncMode_701_; lean_object* v_asyncMode_702_; lean_object* v___x_703_; lean_object* v_base_704_; lean_object* v___x_705_; lean_object* v___x_706_; lean_object* v_simpTheorems_707_; lean_object* v_simprocs_708_; lean_object* v___x_709_; lean_object* v___x_710_; 
v___x_692_ = lean_st_ref_get(v___y_690_);
v_ext_693_ = lean_ctor_get(v_ext_687_, 1);
v_toEnvExtension_694_ = lean_ctor_get(v_ext_693_, 0);
v_ext_695_ = lean_ctor_get(v_simpExt_688_, 1);
v_toEnvExtension_696_ = lean_ctor_get(v_ext_695_, 0);
v_ext_697_ = lean_ctor_get(v_simprocExt_689_, 1);
v_toEnvExtension_698_ = lean_ctor_get(v_ext_697_, 0);
v_env_699_ = lean_ctor_get(v___x_692_, 0);
lean_inc_ref_n(v_env_699_, 3);
lean_dec(v___x_692_);
v_asyncMode_700_ = lean_ctor_get(v_toEnvExtension_694_, 2);
v_asyncMode_701_ = lean_ctor_get(v_toEnvExtension_696_, 2);
v_asyncMode_702_ = lean_ctor_get(v_toEnvExtension_698_, 2);
v___x_703_ = lp_aesop_Aesop_instInhabitedBaseRuleSet_default;
v_base_704_ = l_Lean_ScopedEnvExtension_getState___redArg(v___x_703_, v_ext_687_, v_env_699_, v_asyncMode_700_);
v___x_705_ = l_Lean_Meta_instInhabitedSimpTheorems_default;
v___x_706_ = l_Lean_Meta_Simp_instInhabitedSimprocs_default;
v_simpTheorems_707_ = l_Lean_ScopedEnvExtension_getState___redArg(v___x_705_, v_simpExt_688_, v_env_699_, v_asyncMode_701_);
v_simprocs_708_ = l_Lean_ScopedEnvExtension_getState___redArg(v___x_706_, v_simprocExt_689_, v_env_699_, v_asyncMode_702_);
v___x_709_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_709_, 0, v_base_704_);
lean_ctor_set(v___x_709_, 1, v_simpTheorems_707_);
lean_ctor_set(v___x_709_, 2, v_simprocs_708_);
v___x_710_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_710_, 0, v___x_709_);
return v___x_710_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_getGlobalRuleSetFromData___at___00Aesop_Frontend_getGlobalRuleSet_spec__1___redArg___boxed(lean_object* v_ext_711_, lean_object* v_simpExt_712_, lean_object* v_simprocExt_713_, lean_object* v___y_714_, lean_object* v___y_715_){
_start:
{
lean_object* v_res_716_; 
v_res_716_ = lp_aesop_Aesop_Frontend_getGlobalRuleSetFromData___at___00Aesop_Frontend_getGlobalRuleSet_spec__1___redArg(v_ext_711_, v_simpExt_712_, v_simprocExt_713_, v___y_714_);
lean_dec(v___y_714_);
lean_dec_ref(v_simprocExt_713_);
lean_dec_ref(v_simpExt_712_);
lean_dec_ref(v_ext_711_);
return v_res_716_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_getGlobalRuleSetFromData___at___00Aesop_Frontend_getGlobalRuleSet_spec__1(lean_object* v_ext_717_, lean_object* v_simpExt_718_, lean_object* v_simprocExt_719_, lean_object* v___y_720_, lean_object* v___y_721_){
_start:
{
lean_object* v___x_723_; 
v___x_723_ = lp_aesop_Aesop_Frontend_getGlobalRuleSetFromData___at___00Aesop_Frontend_getGlobalRuleSet_spec__1___redArg(v_ext_717_, v_simpExt_718_, v_simprocExt_719_, v___y_721_);
return v___x_723_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_getGlobalRuleSetFromData___at___00Aesop_Frontend_getGlobalRuleSet_spec__1___boxed(lean_object* v_ext_724_, lean_object* v_simpExt_725_, lean_object* v_simprocExt_726_, lean_object* v___y_727_, lean_object* v___y_728_, lean_object* v___y_729_){
_start:
{
lean_object* v_res_730_; 
v_res_730_ = lp_aesop_Aesop_Frontend_getGlobalRuleSetFromData___at___00Aesop_Frontend_getGlobalRuleSet_spec__1(v_ext_724_, v_simpExt_725_, v_simprocExt_726_, v___y_727_, v___y_728_);
lean_dec(v___y_728_);
lean_dec_ref(v___y_727_);
lean_dec_ref(v_simprocExt_726_);
lean_dec_ref(v_simpExt_725_);
lean_dec_ref(v_ext_724_);
return v_res_730_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_getGlobalRuleSet_spec__0_spec__0_spec__2___redArg(lean_object* v_a_731_, lean_object* v_x_732_){
_start:
{
if (lean_obj_tag(v_x_732_) == 0)
{
lean_object* v___x_733_; 
v___x_733_ = lean_box(0);
return v___x_733_;
}
else
{
lean_object* v_key_734_; lean_object* v_value_735_; lean_object* v_tail_736_; uint8_t v___x_737_; 
v_key_734_ = lean_ctor_get(v_x_732_, 0);
v_value_735_ = lean_ctor_get(v_x_732_, 1);
v_tail_736_ = lean_ctor_get(v_x_732_, 2);
v___x_737_ = lean_name_eq(v_key_734_, v_a_731_);
if (v___x_737_ == 0)
{
v_x_732_ = v_tail_736_;
goto _start;
}
else
{
lean_object* v___x_739_; 
lean_inc(v_value_735_);
v___x_739_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_739_, 0, v_value_735_);
return v___x_739_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_getGlobalRuleSet_spec__0_spec__0_spec__2___redArg___boxed(lean_object* v_a_740_, lean_object* v_x_741_){
_start:
{
lean_object* v_res_742_; 
v_res_742_ = lp_aesop_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_getGlobalRuleSet_spec__0_spec__0_spec__2___redArg(v_a_740_, v_x_741_);
lean_dec(v_x_741_);
lean_dec(v_a_740_);
return v_res_742_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_getGlobalRuleSet_spec__0_spec__0___redArg(lean_object* v_m_743_, lean_object* v_a_744_){
_start:
{
lean_object* v_buckets_745_; lean_object* v___x_746_; uint64_t v___y_748_; 
v_buckets_745_ = lean_ctor_get(v_m_743_, 1);
v___x_746_ = lean_array_get_size(v_buckets_745_);
if (lean_obj_tag(v_a_744_) == 0)
{
uint64_t v___x_762_; 
v___x_762_ = 1723ULL;
v___y_748_ = v___x_762_;
goto v___jp_747_;
}
else
{
uint64_t v_hash_763_; 
v_hash_763_ = lean_ctor_get_uint64(v_a_744_, sizeof(void*)*2);
v___y_748_ = v_hash_763_;
goto v___jp_747_;
}
v___jp_747_:
{
uint64_t v___x_749_; uint64_t v___x_750_; uint64_t v_fold_751_; uint64_t v___x_752_; uint64_t v___x_753_; uint64_t v___x_754_; size_t v___x_755_; size_t v___x_756_; size_t v___x_757_; size_t v___x_758_; size_t v___x_759_; lean_object* v___x_760_; lean_object* v___x_761_; 
v___x_749_ = 32ULL;
v___x_750_ = lean_uint64_shift_right(v___y_748_, v___x_749_);
v_fold_751_ = lean_uint64_xor(v___y_748_, v___x_750_);
v___x_752_ = 16ULL;
v___x_753_ = lean_uint64_shift_right(v_fold_751_, v___x_752_);
v___x_754_ = lean_uint64_xor(v_fold_751_, v___x_753_);
v___x_755_ = lean_uint64_to_usize(v___x_754_);
v___x_756_ = lean_usize_of_nat(v___x_746_);
v___x_757_ = ((size_t)1ULL);
v___x_758_ = lean_usize_sub(v___x_756_, v___x_757_);
v___x_759_ = lean_usize_land(v___x_755_, v___x_758_);
v___x_760_ = lean_array_uget_borrowed(v_buckets_745_, v___x_759_);
v___x_761_ = lp_aesop_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_getGlobalRuleSet_spec__0_spec__0_spec__2___redArg(v_a_744_, v___x_760_);
return v___x_761_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_getGlobalRuleSet_spec__0_spec__0___redArg___boxed(lean_object* v_m_764_, lean_object* v_a_765_){
_start:
{
lean_object* v_res_766_; 
v_res_766_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_getGlobalRuleSet_spec__0_spec__0___redArg(v_m_764_, v_a_765_);
lean_dec(v_a_765_);
lean_dec_ref(v_m_764_);
return v_res_766_;
}
}
static lean_object* _init_lp_aesop_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_getGlobalRuleSet_spec__0_spec__1_spec__4___closed__0(void){
_start:
{
lean_object* v___x_767_; 
v___x_767_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_767_;
}
}
static lean_object* _init_lp_aesop_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_getGlobalRuleSet_spec__0_spec__1_spec__4___closed__1(void){
_start:
{
lean_object* v___x_768_; lean_object* v___x_769_; 
v___x_768_ = lean_obj_once(&lp_aesop_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_getGlobalRuleSet_spec__0_spec__1_spec__4___closed__0, &lp_aesop_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_getGlobalRuleSet_spec__0_spec__1_spec__4___closed__0_once, _init_lp_aesop_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_getGlobalRuleSet_spec__0_spec__1_spec__4___closed__0);
v___x_769_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_769_, 0, v___x_768_);
return v___x_769_;
}
}
static lean_object* _init_lp_aesop_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_getGlobalRuleSet_spec__0_spec__1_spec__4___closed__2(void){
_start:
{
lean_object* v___x_770_; lean_object* v___x_771_; lean_object* v___x_772_; 
v___x_770_ = lean_obj_once(&lp_aesop_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_getGlobalRuleSet_spec__0_spec__1_spec__4___closed__1, &lp_aesop_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_getGlobalRuleSet_spec__0_spec__1_spec__4___closed__1_once, _init_lp_aesop_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_getGlobalRuleSet_spec__0_spec__1_spec__4___closed__1);
v___x_771_ = lean_unsigned_to_nat(0u);
v___x_772_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v___x_772_, 0, v___x_771_);
lean_ctor_set(v___x_772_, 1, v___x_771_);
lean_ctor_set(v___x_772_, 2, v___x_771_);
lean_ctor_set(v___x_772_, 3, v___x_771_);
lean_ctor_set(v___x_772_, 4, v___x_770_);
lean_ctor_set(v___x_772_, 5, v___x_770_);
lean_ctor_set(v___x_772_, 6, v___x_770_);
lean_ctor_set(v___x_772_, 7, v___x_770_);
lean_ctor_set(v___x_772_, 8, v___x_770_);
lean_ctor_set(v___x_772_, 9, v___x_770_);
return v___x_772_;
}
}
static lean_object* _init_lp_aesop_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_getGlobalRuleSet_spec__0_spec__1_spec__4___closed__3(void){
_start:
{
lean_object* v___x_773_; lean_object* v___x_774_; lean_object* v___x_775_; 
v___x_773_ = lean_unsigned_to_nat(32u);
v___x_774_ = lean_mk_empty_array_with_capacity(v___x_773_);
v___x_775_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_775_, 0, v___x_774_);
return v___x_775_;
}
}
static lean_object* _init_lp_aesop_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_getGlobalRuleSet_spec__0_spec__1_spec__4___closed__4(void){
_start:
{
size_t v___x_776_; lean_object* v___x_777_; lean_object* v___x_778_; lean_object* v___x_779_; lean_object* v___x_780_; lean_object* v___x_781_; 
v___x_776_ = ((size_t)5ULL);
v___x_777_ = lean_unsigned_to_nat(0u);
v___x_778_ = lean_unsigned_to_nat(32u);
v___x_779_ = lean_mk_empty_array_with_capacity(v___x_778_);
v___x_780_ = lean_obj_once(&lp_aesop_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_getGlobalRuleSet_spec__0_spec__1_spec__4___closed__3, &lp_aesop_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_getGlobalRuleSet_spec__0_spec__1_spec__4___closed__3_once, _init_lp_aesop_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_getGlobalRuleSet_spec__0_spec__1_spec__4___closed__3);
v___x_781_ = lean_alloc_ctor(0, 4, sizeof(size_t)*1);
lean_ctor_set(v___x_781_, 0, v___x_780_);
lean_ctor_set(v___x_781_, 1, v___x_779_);
lean_ctor_set(v___x_781_, 2, v___x_777_);
lean_ctor_set(v___x_781_, 3, v___x_777_);
lean_ctor_set_usize(v___x_781_, 4, v___x_776_);
return v___x_781_;
}
}
static lean_object* _init_lp_aesop_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_getGlobalRuleSet_spec__0_spec__1_spec__4___closed__5(void){
_start:
{
lean_object* v___x_782_; lean_object* v___x_783_; lean_object* v___x_784_; lean_object* v___x_785_; 
v___x_782_ = lean_box(1);
v___x_783_ = lean_obj_once(&lp_aesop_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_getGlobalRuleSet_spec__0_spec__1_spec__4___closed__4, &lp_aesop_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_getGlobalRuleSet_spec__0_spec__1_spec__4___closed__4_once, _init_lp_aesop_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_getGlobalRuleSet_spec__0_spec__1_spec__4___closed__4);
v___x_784_ = lean_obj_once(&lp_aesop_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_getGlobalRuleSet_spec__0_spec__1_spec__4___closed__1, &lp_aesop_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_getGlobalRuleSet_spec__0_spec__1_spec__4___closed__1_once, _init_lp_aesop_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_getGlobalRuleSet_spec__0_spec__1_spec__4___closed__1);
v___x_785_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_785_, 0, v___x_784_);
lean_ctor_set(v___x_785_, 1, v___x_783_);
lean_ctor_set(v___x_785_, 2, v___x_782_);
return v___x_785_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_getGlobalRuleSet_spec__0_spec__1_spec__4(lean_object* v_msgData_786_, lean_object* v___y_787_, lean_object* v___y_788_){
_start:
{
lean_object* v___x_790_; lean_object* v_env_791_; lean_object* v_options_792_; lean_object* v___x_793_; lean_object* v___x_794_; lean_object* v___x_795_; lean_object* v___x_796_; lean_object* v___x_797_; 
v___x_790_ = lean_st_ref_get(v___y_788_);
v_env_791_ = lean_ctor_get(v___x_790_, 0);
lean_inc_ref(v_env_791_);
lean_dec(v___x_790_);
v_options_792_ = lean_ctor_get(v___y_787_, 2);
v___x_793_ = lean_obj_once(&lp_aesop_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_getGlobalRuleSet_spec__0_spec__1_spec__4___closed__2, &lp_aesop_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_getGlobalRuleSet_spec__0_spec__1_spec__4___closed__2_once, _init_lp_aesop_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_getGlobalRuleSet_spec__0_spec__1_spec__4___closed__2);
v___x_794_ = lean_obj_once(&lp_aesop_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_getGlobalRuleSet_spec__0_spec__1_spec__4___closed__5, &lp_aesop_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_getGlobalRuleSet_spec__0_spec__1_spec__4___closed__5_once, _init_lp_aesop_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_getGlobalRuleSet_spec__0_spec__1_spec__4___closed__5);
lean_inc_ref(v_options_792_);
v___x_795_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_795_, 0, v_env_791_);
lean_ctor_set(v___x_795_, 1, v___x_793_);
lean_ctor_set(v___x_795_, 2, v___x_794_);
lean_ctor_set(v___x_795_, 3, v_options_792_);
v___x_796_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_796_, 0, v___x_795_);
lean_ctor_set(v___x_796_, 1, v_msgData_786_);
v___x_797_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_797_, 0, v___x_796_);
return v___x_797_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_getGlobalRuleSet_spec__0_spec__1_spec__4___boxed(lean_object* v_msgData_798_, lean_object* v___y_799_, lean_object* v___y_800_, lean_object* v___y_801_){
_start:
{
lean_object* v_res_802_; 
v_res_802_ = lp_aesop_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_getGlobalRuleSet_spec__0_spec__1_spec__4(v_msgData_798_, v___y_799_, v___y_800_);
lean_dec(v___y_800_);
lean_dec_ref(v___y_799_);
return v_res_802_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_getGlobalRuleSet_spec__0_spec__1___redArg(lean_object* v_msg_803_, lean_object* v___y_804_, lean_object* v___y_805_){
_start:
{
lean_object* v_ref_807_; lean_object* v___x_808_; lean_object* v_a_809_; lean_object* v___x_811_; uint8_t v_isShared_812_; uint8_t v_isSharedCheck_817_; 
v_ref_807_ = lean_ctor_get(v___y_804_, 5);
v___x_808_ = lp_aesop_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_getGlobalRuleSet_spec__0_spec__1_spec__4(v_msg_803_, v___y_804_, v___y_805_);
v_a_809_ = lean_ctor_get(v___x_808_, 0);
v_isSharedCheck_817_ = !lean_is_exclusive(v___x_808_);
if (v_isSharedCheck_817_ == 0)
{
v___x_811_ = v___x_808_;
v_isShared_812_ = v_isSharedCheck_817_;
goto v_resetjp_810_;
}
else
{
lean_inc(v_a_809_);
lean_dec(v___x_808_);
v___x_811_ = lean_box(0);
v_isShared_812_ = v_isSharedCheck_817_;
goto v_resetjp_810_;
}
v_resetjp_810_:
{
lean_object* v___x_813_; lean_object* v___x_815_; 
lean_inc(v_ref_807_);
v___x_813_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_813_, 0, v_ref_807_);
lean_ctor_set(v___x_813_, 1, v_a_809_);
if (v_isShared_812_ == 0)
{
lean_ctor_set_tag(v___x_811_, 1);
lean_ctor_set(v___x_811_, 0, v___x_813_);
v___x_815_ = v___x_811_;
goto v_reusejp_814_;
}
else
{
lean_object* v_reuseFailAlloc_816_; 
v_reuseFailAlloc_816_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_816_, 0, v___x_813_);
v___x_815_ = v_reuseFailAlloc_816_;
goto v_reusejp_814_;
}
v_reusejp_814_:
{
return v___x_815_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_getGlobalRuleSet_spec__0_spec__1___redArg___boxed(lean_object* v_msg_818_, lean_object* v___y_819_, lean_object* v___y_820_, lean_object* v___y_821_){
_start:
{
lean_object* v_res_822_; 
v_res_822_ = lp_aesop_Lean_throwError___at___00Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_getGlobalRuleSet_spec__0_spec__1___redArg(v_msg_818_, v___y_819_, v___y_820_);
lean_dec(v___y_820_);
lean_dec_ref(v___y_819_);
return v_res_822_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_getGlobalRuleSet_spec__0(lean_object* v_rsName_823_, lean_object* v___y_824_, lean_object* v___y_825_){
_start:
{
lean_object* v___x_827_; 
v___x_827_ = lp_aesop_Aesop_getDeclaredRuleSets();
if (lean_obj_tag(v___x_827_) == 0)
{
lean_object* v_a_828_; lean_object* v___x_829_; 
v_a_828_ = lean_ctor_get(v___x_827_, 0);
lean_inc(v_a_828_);
lean_dec_ref_known(v___x_827_, 1);
v___x_829_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_getGlobalRuleSet_spec__0_spec__0___redArg(v_a_828_, v_rsName_823_);
lean_dec(v_a_828_);
if (lean_obj_tag(v___x_829_) == 1)
{
lean_object* v_val_830_; lean_object* v_snd_831_; lean_object* v_fst_832_; lean_object* v___x_834_; uint8_t v_isShared_835_; uint8_t v_isSharedCheck_902_; 
lean_dec(v_rsName_823_);
v_val_830_ = lean_ctor_get(v___x_829_, 0);
lean_inc(v_val_830_);
lean_dec_ref_known(v___x_829_, 1);
v_snd_831_ = lean_ctor_get(v_val_830_, 1);
v_fst_832_ = lean_ctor_get(v_val_830_, 0);
v_isSharedCheck_902_ = !lean_is_exclusive(v_val_830_);
if (v_isSharedCheck_902_ == 0)
{
v___x_834_ = v_val_830_;
v_isShared_835_ = v_isSharedCheck_902_;
goto v_resetjp_833_;
}
else
{
lean_inc(v_snd_831_);
lean_inc(v_fst_832_);
lean_dec(v_val_830_);
v___x_834_ = lean_box(0);
v_isShared_835_ = v_isSharedCheck_902_;
goto v_resetjp_833_;
}
v_resetjp_833_:
{
lean_object* v_fst_836_; lean_object* v_snd_837_; lean_object* v___x_839_; uint8_t v_isShared_840_; uint8_t v_isSharedCheck_901_; 
v_fst_836_ = lean_ctor_get(v_snd_831_, 0);
v_snd_837_ = lean_ctor_get(v_snd_831_, 1);
v_isSharedCheck_901_ = !lean_is_exclusive(v_snd_831_);
if (v_isSharedCheck_901_ == 0)
{
v___x_839_ = v_snd_831_;
v_isShared_840_ = v_isSharedCheck_901_;
goto v_resetjp_838_;
}
else
{
lean_inc(v_snd_837_);
lean_inc(v_fst_836_);
lean_dec(v_snd_831_);
v___x_839_ = lean_box(0);
v_isShared_840_ = v_isSharedCheck_901_;
goto v_resetjp_838_;
}
v_resetjp_838_:
{
lean_object* v___x_841_; 
v___x_841_ = l_Lean_Meta_getSimpExtension_x3f(v_fst_836_, v___y_824_, v___y_825_);
if (lean_obj_tag(v___x_841_) == 0)
{
lean_object* v_a_842_; 
v_a_842_ = lean_ctor_get(v___x_841_, 0);
lean_inc(v_a_842_);
lean_dec_ref_known(v___x_841_, 1);
if (lean_obj_tag(v_a_842_) == 1)
{
lean_object* v_val_843_; lean_object* v___x_845_; uint8_t v_isShared_846_; uint8_t v_isSharedCheck_886_; 
v_val_843_ = lean_ctor_get(v_a_842_, 0);
v_isSharedCheck_886_ = !lean_is_exclusive(v_a_842_);
if (v_isSharedCheck_886_ == 0)
{
v___x_845_ = v_a_842_;
v_isShared_846_ = v_isSharedCheck_886_;
goto v_resetjp_844_;
}
else
{
lean_inc(v_val_843_);
lean_dec(v_a_842_);
v___x_845_ = lean_box(0);
v_isShared_846_ = v_isSharedCheck_886_;
goto v_resetjp_844_;
}
v_resetjp_844_:
{
lean_object* v___x_847_; 
lean_inc(v_fst_836_);
v___x_847_ = l_Lean_Meta_Simp_getSimprocExtension_x3f(v_fst_836_);
if (lean_obj_tag(v___x_847_) == 0)
{
lean_object* v_a_848_; lean_object* v___x_850_; uint8_t v_isShared_851_; uint8_t v_isSharedCheck_870_; 
lean_del_object(v___x_845_);
v_a_848_ = lean_ctor_get(v___x_847_, 0);
v_isSharedCheck_870_ = !lean_is_exclusive(v___x_847_);
if (v_isSharedCheck_870_ == 0)
{
v___x_850_ = v___x_847_;
v_isShared_851_ = v_isSharedCheck_870_;
goto v_resetjp_849_;
}
else
{
lean_inc(v_a_848_);
lean_dec(v___x_847_);
v___x_850_ = lean_box(0);
v_isShared_851_ = v_isSharedCheck_870_;
goto v_resetjp_849_;
}
v_resetjp_849_:
{
if (lean_obj_tag(v_a_848_) == 1)
{
lean_object* v_val_852_; lean_object* v___x_854_; 
v_val_852_ = lean_ctor_get(v_a_848_, 0);
lean_inc(v_val_852_);
lean_dec_ref_known(v_a_848_, 1);
if (v_isShared_840_ == 0)
{
lean_ctor_set(v___x_839_, 1, v_val_852_);
lean_ctor_set(v___x_839_, 0, v_snd_837_);
v___x_854_ = v___x_839_;
goto v_reusejp_853_;
}
else
{
lean_object* v_reuseFailAlloc_863_; 
v_reuseFailAlloc_863_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_863_, 0, v_snd_837_);
lean_ctor_set(v_reuseFailAlloc_863_, 1, v_val_852_);
v___x_854_ = v_reuseFailAlloc_863_;
goto v_reusejp_853_;
}
v_reusejp_853_:
{
lean_object* v___x_856_; 
if (v_isShared_835_ == 0)
{
lean_ctor_set(v___x_834_, 1, v___x_854_);
lean_ctor_set(v___x_834_, 0, v_val_843_);
v___x_856_ = v___x_834_;
goto v_reusejp_855_;
}
else
{
lean_object* v_reuseFailAlloc_862_; 
v_reuseFailAlloc_862_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_862_, 0, v_val_843_);
lean_ctor_set(v_reuseFailAlloc_862_, 1, v___x_854_);
v___x_856_ = v_reuseFailAlloc_862_;
goto v_reusejp_855_;
}
v_reusejp_855_:
{
lean_object* v___x_857_; lean_object* v___x_858_; lean_object* v___x_860_; 
v___x_857_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_857_, 0, v_fst_836_);
lean_ctor_set(v___x_857_, 1, v___x_856_);
v___x_858_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_858_, 0, v_fst_832_);
lean_ctor_set(v___x_858_, 1, v___x_857_);
if (v_isShared_851_ == 0)
{
lean_ctor_set(v___x_850_, 0, v___x_858_);
v___x_860_ = v___x_850_;
goto v_reusejp_859_;
}
else
{
lean_object* v_reuseFailAlloc_861_; 
v_reuseFailAlloc_861_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_861_, 0, v___x_858_);
v___x_860_ = v_reuseFailAlloc_861_;
goto v_reusejp_859_;
}
v_reusejp_859_:
{
return v___x_860_;
}
}
}
}
else
{
lean_object* v___x_864_; lean_object* v___x_865_; lean_object* v___x_866_; lean_object* v___x_867_; lean_object* v___x_868_; lean_object* v___x_869_; 
lean_del_object(v___x_850_);
lean_dec(v_a_848_);
lean_dec(v_val_843_);
lean_del_object(v___x_839_);
lean_dec(v_snd_837_);
lean_del_object(v___x_834_);
lean_dec(v_fst_832_);
v___x_864_ = lean_obj_once(&lp_aesop_Aesop_Frontend_getGlobalRuleSetData___redArg___lam__0___closed__1, &lp_aesop_Aesop_Frontend_getGlobalRuleSetData___redArg___lam__0___closed__1_once, _init_lp_aesop_Aesop_Frontend_getGlobalRuleSetData___redArg___lam__0___closed__1);
v___x_865_ = l_Lean_MessageData_ofName(v_fst_836_);
v___x_866_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_866_, 0, v___x_864_);
lean_ctor_set(v___x_866_, 1, v___x_865_);
v___x_867_ = lean_obj_once(&lp_aesop_Aesop_Frontend_getGlobalRuleSetData___redArg___lam__0___closed__3, &lp_aesop_Aesop_Frontend_getGlobalRuleSetData___redArg___lam__0___closed__3_once, _init_lp_aesop_Aesop_Frontend_getGlobalRuleSetData___redArg___lam__0___closed__3);
v___x_868_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_868_, 0, v___x_866_);
lean_ctor_set(v___x_868_, 1, v___x_867_);
v___x_869_ = lp_aesop_Lean_throwError___at___00Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_getGlobalRuleSet_spec__0_spec__1___redArg(v___x_868_, v___y_824_, v___y_825_);
return v___x_869_;
}
}
}
else
{
lean_object* v_a_871_; lean_object* v___x_873_; uint8_t v_isShared_874_; uint8_t v_isSharedCheck_885_; 
lean_dec(v_val_843_);
lean_del_object(v___x_839_);
lean_dec(v_snd_837_);
lean_dec(v_fst_836_);
lean_del_object(v___x_834_);
lean_dec(v_fst_832_);
v_a_871_ = lean_ctor_get(v___x_847_, 0);
v_isSharedCheck_885_ = !lean_is_exclusive(v___x_847_);
if (v_isSharedCheck_885_ == 0)
{
v___x_873_ = v___x_847_;
v_isShared_874_ = v_isSharedCheck_885_;
goto v_resetjp_872_;
}
else
{
lean_inc(v_a_871_);
lean_dec(v___x_847_);
v___x_873_ = lean_box(0);
v_isShared_874_ = v_isSharedCheck_885_;
goto v_resetjp_872_;
}
v_resetjp_872_:
{
lean_object* v_ref_875_; lean_object* v___x_876_; lean_object* v___x_878_; 
v_ref_875_ = lean_ctor_get(v___y_824_, 5);
v___x_876_ = lean_io_error_to_string(v_a_871_);
if (v_isShared_846_ == 0)
{
lean_ctor_set_tag(v___x_845_, 3);
lean_ctor_set(v___x_845_, 0, v___x_876_);
v___x_878_ = v___x_845_;
goto v_reusejp_877_;
}
else
{
lean_object* v_reuseFailAlloc_884_; 
v_reuseFailAlloc_884_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v_reuseFailAlloc_884_, 0, v___x_876_);
v___x_878_ = v_reuseFailAlloc_884_;
goto v_reusejp_877_;
}
v_reusejp_877_:
{
lean_object* v___x_879_; lean_object* v___x_880_; lean_object* v___x_882_; 
v___x_879_ = l_Lean_MessageData_ofFormat(v___x_878_);
lean_inc(v_ref_875_);
v___x_880_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_880_, 0, v_ref_875_);
lean_ctor_set(v___x_880_, 1, v___x_879_);
if (v_isShared_874_ == 0)
{
lean_ctor_set(v___x_873_, 0, v___x_880_);
v___x_882_ = v___x_873_;
goto v_reusejp_881_;
}
else
{
lean_object* v_reuseFailAlloc_883_; 
v_reuseFailAlloc_883_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_883_, 0, v___x_880_);
v___x_882_ = v_reuseFailAlloc_883_;
goto v_reusejp_881_;
}
v_reusejp_881_:
{
return v___x_882_;
}
}
}
}
}
}
else
{
lean_object* v___x_887_; lean_object* v___x_888_; lean_object* v___x_889_; lean_object* v___x_890_; lean_object* v___x_891_; lean_object* v___x_892_; 
lean_dec(v_a_842_);
lean_del_object(v___x_839_);
lean_dec(v_snd_837_);
lean_del_object(v___x_834_);
lean_dec(v_fst_832_);
v___x_887_ = lean_obj_once(&lp_aesop_Aesop_Frontend_getGlobalRuleSetData___redArg___lam__0___closed__1, &lp_aesop_Aesop_Frontend_getGlobalRuleSetData___redArg___lam__0___closed__1_once, _init_lp_aesop_Aesop_Frontend_getGlobalRuleSetData___redArg___lam__0___closed__1);
v___x_888_ = l_Lean_MessageData_ofName(v_fst_836_);
v___x_889_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_889_, 0, v___x_887_);
lean_ctor_set(v___x_889_, 1, v___x_888_);
v___x_890_ = lean_obj_once(&lp_aesop_Aesop_Frontend_getGlobalRuleSetData___redArg___lam__0___closed__3, &lp_aesop_Aesop_Frontend_getGlobalRuleSetData___redArg___lam__0___closed__3_once, _init_lp_aesop_Aesop_Frontend_getGlobalRuleSetData___redArg___lam__0___closed__3);
v___x_891_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_891_, 0, v___x_889_);
lean_ctor_set(v___x_891_, 1, v___x_890_);
v___x_892_ = lp_aesop_Lean_throwError___at___00Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_getGlobalRuleSet_spec__0_spec__1___redArg(v___x_891_, v___y_824_, v___y_825_);
return v___x_892_;
}
}
else
{
lean_object* v_a_893_; lean_object* v___x_895_; uint8_t v_isShared_896_; uint8_t v_isSharedCheck_900_; 
lean_del_object(v___x_839_);
lean_dec(v_snd_837_);
lean_dec(v_fst_836_);
lean_del_object(v___x_834_);
lean_dec(v_fst_832_);
v_a_893_ = lean_ctor_get(v___x_841_, 0);
v_isSharedCheck_900_ = !lean_is_exclusive(v___x_841_);
if (v_isSharedCheck_900_ == 0)
{
v___x_895_ = v___x_841_;
v_isShared_896_ = v_isSharedCheck_900_;
goto v_resetjp_894_;
}
else
{
lean_inc(v_a_893_);
lean_dec(v___x_841_);
v___x_895_ = lean_box(0);
v_isShared_896_ = v_isSharedCheck_900_;
goto v_resetjp_894_;
}
v_resetjp_894_:
{
lean_object* v___x_898_; 
if (v_isShared_896_ == 0)
{
v___x_898_ = v___x_895_;
goto v_reusejp_897_;
}
else
{
lean_object* v_reuseFailAlloc_899_; 
v_reuseFailAlloc_899_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_899_, 0, v_a_893_);
v___x_898_ = v_reuseFailAlloc_899_;
goto v_reusejp_897_;
}
v_reusejp_897_:
{
return v___x_898_;
}
}
}
}
}
}
else
{
lean_object* v___x_903_; lean_object* v___x_904_; lean_object* v___x_905_; lean_object* v___x_906_; lean_object* v___x_907_; lean_object* v___x_908_; 
lean_dec(v___x_829_);
v___x_903_ = lean_obj_once(&lp_aesop_Aesop_Frontend_getGlobalRuleSetData___redArg___lam__2___closed__1, &lp_aesop_Aesop_Frontend_getGlobalRuleSetData___redArg___lam__2___closed__1_once, _init_lp_aesop_Aesop_Frontend_getGlobalRuleSetData___redArg___lam__2___closed__1);
v___x_904_ = l_Lean_MessageData_ofName(v_rsName_823_);
v___x_905_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_905_, 0, v___x_903_);
lean_ctor_set(v___x_905_, 1, v___x_904_);
v___x_906_ = lean_obj_once(&lp_aesop_Aesop_Frontend_getGlobalRuleSetData___redArg___lam__2___closed__3, &lp_aesop_Aesop_Frontend_getGlobalRuleSetData___redArg___lam__2___closed__3_once, _init_lp_aesop_Aesop_Frontend_getGlobalRuleSetData___redArg___lam__2___closed__3);
v___x_907_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_907_, 0, v___x_905_);
lean_ctor_set(v___x_907_, 1, v___x_906_);
v___x_908_ = lp_aesop_Lean_throwError___at___00Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_getGlobalRuleSet_spec__0_spec__1___redArg(v___x_907_, v___y_824_, v___y_825_);
return v___x_908_;
}
}
else
{
lean_object* v_a_909_; lean_object* v___x_911_; uint8_t v_isShared_912_; uint8_t v_isSharedCheck_921_; 
lean_dec(v_rsName_823_);
v_a_909_ = lean_ctor_get(v___x_827_, 0);
v_isSharedCheck_921_ = !lean_is_exclusive(v___x_827_);
if (v_isSharedCheck_921_ == 0)
{
v___x_911_ = v___x_827_;
v_isShared_912_ = v_isSharedCheck_921_;
goto v_resetjp_910_;
}
else
{
lean_inc(v_a_909_);
lean_dec(v___x_827_);
v___x_911_ = lean_box(0);
v_isShared_912_ = v_isSharedCheck_921_;
goto v_resetjp_910_;
}
v_resetjp_910_:
{
lean_object* v_ref_913_; lean_object* v___x_914_; lean_object* v___x_915_; lean_object* v___x_916_; lean_object* v___x_917_; lean_object* v___x_919_; 
v_ref_913_ = lean_ctor_get(v___y_824_, 5);
v___x_914_ = lean_io_error_to_string(v_a_909_);
v___x_915_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_915_, 0, v___x_914_);
v___x_916_ = l_Lean_MessageData_ofFormat(v___x_915_);
lean_inc(v_ref_913_);
v___x_917_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_917_, 0, v_ref_913_);
lean_ctor_set(v___x_917_, 1, v___x_916_);
if (v_isShared_912_ == 0)
{
lean_ctor_set(v___x_911_, 0, v___x_917_);
v___x_919_ = v___x_911_;
goto v_reusejp_918_;
}
else
{
lean_object* v_reuseFailAlloc_920_; 
v_reuseFailAlloc_920_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_920_, 0, v___x_917_);
v___x_919_ = v_reuseFailAlloc_920_;
goto v_reusejp_918_;
}
v_reusejp_918_:
{
return v___x_919_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_getGlobalRuleSet_spec__0___boxed(lean_object* v_rsName_922_, lean_object* v___y_923_, lean_object* v___y_924_, lean_object* v___y_925_){
_start:
{
lean_object* v_res_926_; 
v_res_926_ = lp_aesop_Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_getGlobalRuleSet_spec__0(v_rsName_922_, v___y_923_, v___y_924_);
lean_dec(v___y_924_);
lean_dec_ref(v___y_923_);
return v_res_926_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_getGlobalRuleSet(lean_object* v_rsName_927_, lean_object* v_a_928_, lean_object* v_a_929_){
_start:
{
lean_object* v___x_931_; 
v___x_931_ = lp_aesop_Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_getGlobalRuleSet_spec__0(v_rsName_927_, v_a_928_, v_a_929_);
if (lean_obj_tag(v___x_931_) == 0)
{
lean_object* v_a_932_; lean_object* v_snd_933_; lean_object* v_snd_934_; lean_object* v_snd_935_; lean_object* v_fst_936_; lean_object* v_fst_937_; lean_object* v_fst_938_; lean_object* v___x_940_; uint8_t v_isShared_941_; uint8_t v_isSharedCheck_963_; 
v_a_932_ = lean_ctor_get(v___x_931_, 0);
lean_inc(v_a_932_);
lean_dec_ref_known(v___x_931_, 1);
v_snd_933_ = lean_ctor_get(v_a_932_, 1);
lean_inc(v_snd_933_);
v_snd_934_ = lean_ctor_get(v_snd_933_, 1);
lean_inc(v_snd_934_);
v_snd_935_ = lean_ctor_get(v_snd_934_, 1);
lean_inc(v_snd_935_);
v_fst_936_ = lean_ctor_get(v_a_932_, 0);
lean_inc(v_fst_936_);
lean_dec(v_a_932_);
v_fst_937_ = lean_ctor_get(v_snd_933_, 0);
lean_inc(v_fst_937_);
lean_dec(v_snd_933_);
v_fst_938_ = lean_ctor_get(v_snd_934_, 0);
v_isSharedCheck_963_ = !lean_is_exclusive(v_snd_934_);
if (v_isSharedCheck_963_ == 0)
{
lean_object* v_unused_964_; 
v_unused_964_ = lean_ctor_get(v_snd_934_, 1);
lean_dec(v_unused_964_);
v___x_940_ = v_snd_934_;
v_isShared_941_ = v_isSharedCheck_963_;
goto v_resetjp_939_;
}
else
{
lean_inc(v_fst_938_);
lean_dec(v_snd_934_);
v___x_940_ = lean_box(0);
v_isShared_941_ = v_isSharedCheck_963_;
goto v_resetjp_939_;
}
v_resetjp_939_:
{
lean_object* v_fst_942_; lean_object* v_snd_943_; lean_object* v___x_945_; uint8_t v_isShared_946_; uint8_t v_isSharedCheck_962_; 
v_fst_942_ = lean_ctor_get(v_snd_935_, 0);
v_snd_943_ = lean_ctor_get(v_snd_935_, 1);
v_isSharedCheck_962_ = !lean_is_exclusive(v_snd_935_);
if (v_isSharedCheck_962_ == 0)
{
v___x_945_ = v_snd_935_;
v_isShared_946_ = v_isSharedCheck_962_;
goto v_resetjp_944_;
}
else
{
lean_inc(v_snd_943_);
lean_inc(v_fst_942_);
lean_dec(v_snd_935_);
v___x_945_ = lean_box(0);
v_isShared_946_ = v_isSharedCheck_962_;
goto v_resetjp_944_;
}
v_resetjp_944_:
{
lean_object* v___x_947_; lean_object* v_a_948_; lean_object* v___x_950_; uint8_t v_isShared_951_; uint8_t v_isSharedCheck_961_; 
v___x_947_ = lp_aesop_Aesop_Frontend_getGlobalRuleSetFromData___at___00Aesop_Frontend_getGlobalRuleSet_spec__1___redArg(v_fst_936_, v_fst_938_, v_snd_943_, v_a_929_);
lean_dec(v_snd_943_);
lean_dec(v_fst_938_);
lean_dec(v_fst_936_);
v_a_948_ = lean_ctor_get(v___x_947_, 0);
v_isSharedCheck_961_ = !lean_is_exclusive(v___x_947_);
if (v_isSharedCheck_961_ == 0)
{
v___x_950_ = v___x_947_;
v_isShared_951_ = v_isSharedCheck_961_;
goto v_resetjp_949_;
}
else
{
lean_inc(v_a_948_);
lean_dec(v___x_947_);
v___x_950_ = lean_box(0);
v_isShared_951_ = v_isSharedCheck_961_;
goto v_resetjp_949_;
}
v_resetjp_949_:
{
lean_object* v___x_953_; 
if (v_isShared_946_ == 0)
{
lean_ctor_set(v___x_945_, 1, v_fst_942_);
lean_ctor_set(v___x_945_, 0, v_fst_937_);
v___x_953_ = v___x_945_;
goto v_reusejp_952_;
}
else
{
lean_object* v_reuseFailAlloc_960_; 
v_reuseFailAlloc_960_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_960_, 0, v_fst_937_);
lean_ctor_set(v_reuseFailAlloc_960_, 1, v_fst_942_);
v___x_953_ = v_reuseFailAlloc_960_;
goto v_reusejp_952_;
}
v_reusejp_952_:
{
lean_object* v___x_955_; 
if (v_isShared_941_ == 0)
{
lean_ctor_set(v___x_940_, 1, v___x_953_);
lean_ctor_set(v___x_940_, 0, v_a_948_);
v___x_955_ = v___x_940_;
goto v_reusejp_954_;
}
else
{
lean_object* v_reuseFailAlloc_959_; 
v_reuseFailAlloc_959_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_959_, 0, v_a_948_);
lean_ctor_set(v_reuseFailAlloc_959_, 1, v___x_953_);
v___x_955_ = v_reuseFailAlloc_959_;
goto v_reusejp_954_;
}
v_reusejp_954_:
{
lean_object* v___x_957_; 
if (v_isShared_951_ == 0)
{
lean_ctor_set(v___x_950_, 0, v___x_955_);
v___x_957_ = v___x_950_;
goto v_reusejp_956_;
}
else
{
lean_object* v_reuseFailAlloc_958_; 
v_reuseFailAlloc_958_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_958_, 0, v___x_955_);
v___x_957_ = v_reuseFailAlloc_958_;
goto v_reusejp_956_;
}
v_reusejp_956_:
{
return v___x_957_;
}
}
}
}
}
}
}
else
{
lean_object* v_a_965_; lean_object* v___x_967_; uint8_t v_isShared_968_; uint8_t v_isSharedCheck_972_; 
v_a_965_ = lean_ctor_get(v___x_931_, 0);
v_isSharedCheck_972_ = !lean_is_exclusive(v___x_931_);
if (v_isSharedCheck_972_ == 0)
{
v___x_967_ = v___x_931_;
v_isShared_968_ = v_isSharedCheck_972_;
goto v_resetjp_966_;
}
else
{
lean_inc(v_a_965_);
lean_dec(v___x_931_);
v___x_967_ = lean_box(0);
v_isShared_968_ = v_isSharedCheck_972_;
goto v_resetjp_966_;
}
v_resetjp_966_:
{
lean_object* v___x_970_; 
if (v_isShared_968_ == 0)
{
v___x_970_ = v___x_967_;
goto v_reusejp_969_;
}
else
{
lean_object* v_reuseFailAlloc_971_; 
v_reuseFailAlloc_971_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_971_, 0, v_a_965_);
v___x_970_ = v_reuseFailAlloc_971_;
goto v_reusejp_969_;
}
v_reusejp_969_:
{
return v___x_970_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_getGlobalRuleSet___boxed(lean_object* v_rsName_973_, lean_object* v_a_974_, lean_object* v_a_975_, lean_object* v_a_976_){
_start:
{
lean_object* v_res_977_; 
v_res_977_ = lp_aesop_Aesop_Frontend_getGlobalRuleSet(v_rsName_973_, v_a_974_, v_a_975_);
lean_dec(v_a_975_);
lean_dec_ref(v_a_974_);
return v_res_977_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_getGlobalRuleSet_spec__0_spec__0(lean_object* v_00_u03b2_978_, lean_object* v_m_979_, lean_object* v_a_980_){
_start:
{
lean_object* v___x_981_; 
v___x_981_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_getGlobalRuleSet_spec__0_spec__0___redArg(v_m_979_, v_a_980_);
return v___x_981_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_getGlobalRuleSet_spec__0_spec__0___boxed(lean_object* v_00_u03b2_982_, lean_object* v_m_983_, lean_object* v_a_984_){
_start:
{
lean_object* v_res_985_; 
v_res_985_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_getGlobalRuleSet_spec__0_spec__0(v_00_u03b2_982_, v_m_983_, v_a_984_);
lean_dec(v_a_984_);
lean_dec_ref(v_m_983_);
return v_res_985_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_getGlobalRuleSet_spec__0_spec__1(lean_object* v_00_u03b1_986_, lean_object* v_msg_987_, lean_object* v___y_988_, lean_object* v___y_989_){
_start:
{
lean_object* v___x_991_; 
v___x_991_ = lp_aesop_Lean_throwError___at___00Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_getGlobalRuleSet_spec__0_spec__1___redArg(v_msg_987_, v___y_988_, v___y_989_);
return v___x_991_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_getGlobalRuleSet_spec__0_spec__1___boxed(lean_object* v_00_u03b1_992_, lean_object* v_msg_993_, lean_object* v___y_994_, lean_object* v___y_995_, lean_object* v___y_996_){
_start:
{
lean_object* v_res_997_; 
v_res_997_ = lp_aesop_Lean_throwError___at___00Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_getGlobalRuleSet_spec__0_spec__1(v_00_u03b1_992_, v_msg_993_, v___y_994_, v___y_995_);
lean_dec(v___y_995_);
lean_dec_ref(v___y_994_);
return v_res_997_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_getGlobalRuleSet_spec__0_spec__0_spec__2(lean_object* v_00_u03b2_998_, lean_object* v_a_999_, lean_object* v_x_1000_){
_start:
{
lean_object* v___x_1001_; 
v___x_1001_ = lp_aesop_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_getGlobalRuleSet_spec__0_spec__0_spec__2___redArg(v_a_999_, v_x_1000_);
return v___x_1001_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_getGlobalRuleSet_spec__0_spec__0_spec__2___boxed(lean_object* v_00_u03b2_1002_, lean_object* v_a_1003_, lean_object* v_x_1004_){
_start:
{
lean_object* v_res_1005_; 
v_res_1005_ = lp_aesop_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_getGlobalRuleSet_spec__0_spec__0_spec__2(v_00_u03b2_1002_, v_a_1003_, v_x_1004_);
lean_dec(v_x_1004_);
lean_dec(v_a_1003_);
return v_res_1005_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Frontend_getGlobalRuleSets_spec__0(size_t v_sz_1006_, size_t v_i_1007_, lean_object* v_bs_1008_, lean_object* v___y_1009_, lean_object* v___y_1010_){
_start:
{
uint8_t v___x_1012_; 
v___x_1012_ = lean_usize_dec_lt(v_i_1007_, v_sz_1006_);
if (v___x_1012_ == 0)
{
lean_object* v___x_1013_; 
v___x_1013_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1013_, 0, v_bs_1008_);
return v___x_1013_;
}
else
{
lean_object* v_v_1014_; lean_object* v___x_1015_; 
v_v_1014_ = lean_array_uget_borrowed(v_bs_1008_, v_i_1007_);
lean_inc(v_v_1014_);
v___x_1015_ = lp_aesop_Aesop_Frontend_getGlobalRuleSet(v_v_1014_, v___y_1009_, v___y_1010_);
if (lean_obj_tag(v___x_1015_) == 0)
{
lean_object* v_a_1016_; lean_object* v___x_1017_; lean_object* v_bs_x27_1018_; size_t v___x_1019_; size_t v___x_1020_; lean_object* v___x_1021_; 
v_a_1016_ = lean_ctor_get(v___x_1015_, 0);
lean_inc(v_a_1016_);
lean_dec_ref_known(v___x_1015_, 1);
v___x_1017_ = lean_unsigned_to_nat(0u);
v_bs_x27_1018_ = lean_array_uset(v_bs_1008_, v_i_1007_, v___x_1017_);
v___x_1019_ = ((size_t)1ULL);
v___x_1020_ = lean_usize_add(v_i_1007_, v___x_1019_);
v___x_1021_ = lean_array_uset(v_bs_x27_1018_, v_i_1007_, v_a_1016_);
v_i_1007_ = v___x_1020_;
v_bs_1008_ = v___x_1021_;
goto _start;
}
else
{
lean_object* v_a_1023_; lean_object* v___x_1025_; uint8_t v_isShared_1026_; uint8_t v_isSharedCheck_1030_; 
lean_dec_ref(v_bs_1008_);
v_a_1023_ = lean_ctor_get(v___x_1015_, 0);
v_isSharedCheck_1030_ = !lean_is_exclusive(v___x_1015_);
if (v_isSharedCheck_1030_ == 0)
{
v___x_1025_ = v___x_1015_;
v_isShared_1026_ = v_isSharedCheck_1030_;
goto v_resetjp_1024_;
}
else
{
lean_inc(v_a_1023_);
lean_dec(v___x_1015_);
v___x_1025_ = lean_box(0);
v_isShared_1026_ = v_isSharedCheck_1030_;
goto v_resetjp_1024_;
}
v_resetjp_1024_:
{
lean_object* v___x_1028_; 
if (v_isShared_1026_ == 0)
{
v___x_1028_ = v___x_1025_;
goto v_reusejp_1027_;
}
else
{
lean_object* v_reuseFailAlloc_1029_; 
v_reuseFailAlloc_1029_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1029_, 0, v_a_1023_);
v___x_1028_ = v_reuseFailAlloc_1029_;
goto v_reusejp_1027_;
}
v_reusejp_1027_:
{
return v___x_1028_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Frontend_getGlobalRuleSets_spec__0___boxed(lean_object* v_sz_1031_, lean_object* v_i_1032_, lean_object* v_bs_1033_, lean_object* v___y_1034_, lean_object* v___y_1035_, lean_object* v___y_1036_){
_start:
{
size_t v_sz_boxed_1037_; size_t v_i_boxed_1038_; lean_object* v_res_1039_; 
v_sz_boxed_1037_ = lean_unbox_usize(v_sz_1031_);
lean_dec(v_sz_1031_);
v_i_boxed_1038_ = lean_unbox_usize(v_i_1032_);
lean_dec(v_i_1032_);
v_res_1039_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Frontend_getGlobalRuleSets_spec__0(v_sz_boxed_1037_, v_i_boxed_1038_, v_bs_1033_, v___y_1034_, v___y_1035_);
lean_dec(v___y_1035_);
lean_dec_ref(v___y_1034_);
return v_res_1039_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_getGlobalRuleSets(lean_object* v_rsNames_1040_, lean_object* v_a_1041_, lean_object* v_a_1042_){
_start:
{
size_t v_sz_1044_; size_t v___x_1045_; lean_object* v___x_1046_; 
v_sz_1044_ = lean_array_size(v_rsNames_1040_);
v___x_1045_ = ((size_t)0ULL);
v___x_1046_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Frontend_getGlobalRuleSets_spec__0(v_sz_1044_, v___x_1045_, v_rsNames_1040_, v_a_1041_, v_a_1042_);
return v___x_1046_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_getGlobalRuleSets___boxed(lean_object* v_rsNames_1047_, lean_object* v_a_1048_, lean_object* v_a_1049_, lean_object* v_a_1050_){
_start:
{
lean_object* v_res_1051_; 
v_res_1051_ = lp_aesop_Aesop_Frontend_getGlobalRuleSets(v_rsNames_1047_, v_a_1048_, v_a_1049_);
lean_dec(v_a_1049_);
lean_dec_ref(v_a_1048_);
return v_res_1051_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00Aesop_Frontend_getDefaultGlobalRuleSets_spec__0(lean_object* v_x_1052_, lean_object* v_x_1053_){
_start:
{
if (lean_obj_tag(v_x_1053_) == 0)
{
return v_x_1052_;
}
else
{
lean_object* v_key_1054_; lean_object* v_tail_1055_; lean_object* v___x_1056_; 
v_key_1054_ = lean_ctor_get(v_x_1053_, 0);
lean_inc(v_key_1054_);
v_tail_1055_ = lean_ctor_get(v_x_1053_, 2);
lean_inc(v_tail_1055_);
lean_dec_ref_known(v_x_1053_, 3);
v___x_1056_ = lean_array_push(v_x_1052_, v_key_1054_);
v_x_1052_ = v___x_1056_;
v_x_1053_ = v_tail_1055_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Frontend_getDefaultGlobalRuleSets_spec__1(lean_object* v_as_1058_, size_t v_i_1059_, size_t v_stop_1060_, lean_object* v_b_1061_){
_start:
{
uint8_t v___x_1062_; 
v___x_1062_ = lean_usize_dec_eq(v_i_1059_, v_stop_1060_);
if (v___x_1062_ == 0)
{
lean_object* v___x_1063_; lean_object* v___x_1064_; size_t v___x_1065_; size_t v___x_1066_; 
v___x_1063_ = lean_array_uget_borrowed(v_as_1058_, v_i_1059_);
lean_inc(v___x_1063_);
v___x_1064_ = lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00Aesop_Frontend_getDefaultGlobalRuleSets_spec__0(v_b_1061_, v___x_1063_);
v___x_1065_ = ((size_t)1ULL);
v___x_1066_ = lean_usize_add(v_i_1059_, v___x_1065_);
v_i_1059_ = v___x_1066_;
v_b_1061_ = v___x_1064_;
goto _start;
}
else
{
return v_b_1061_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Frontend_getDefaultGlobalRuleSets_spec__1___boxed(lean_object* v_as_1068_, lean_object* v_i_1069_, lean_object* v_stop_1070_, lean_object* v_b_1071_){
_start:
{
size_t v_i_boxed_1072_; size_t v_stop_boxed_1073_; lean_object* v_res_1074_; 
v_i_boxed_1072_ = lean_unbox_usize(v_i_1069_);
lean_dec(v_i_1069_);
v_stop_boxed_1073_ = lean_unbox_usize(v_stop_1070_);
lean_dec(v_stop_1070_);
v_res_1074_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Frontend_getDefaultGlobalRuleSets_spec__1(v_as_1068_, v_i_boxed_1072_, v_stop_boxed_1073_, v_b_1071_);
lean_dec_ref(v_as_1068_);
return v_res_1074_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_getDefaultGlobalRuleSets(lean_object* v_a_1075_, lean_object* v_a_1076_){
_start:
{
lean_object* v___x_1078_; 
v___x_1078_ = lp_aesop_Aesop_getDefaultRuleSetNames();
if (lean_obj_tag(v___x_1078_) == 0)
{
lean_object* v_a_1079_; lean_object* v_size_1080_; lean_object* v_buckets_1081_; lean_object* v___x_1082_; lean_object* v___x_1083_; lean_object* v___x_1084_; uint8_t v___x_1085_; 
v_a_1079_ = lean_ctor_get(v___x_1078_, 0);
lean_inc(v_a_1079_);
lean_dec_ref_known(v___x_1078_, 1);
v_size_1080_ = lean_ctor_get(v_a_1079_, 0);
lean_inc(v_size_1080_);
v_buckets_1081_ = lean_ctor_get(v_a_1079_, 1);
lean_inc_ref(v_buckets_1081_);
lean_dec(v_a_1079_);
v___x_1082_ = lean_mk_empty_array_with_capacity(v_size_1080_);
lean_dec(v_size_1080_);
v___x_1083_ = lean_unsigned_to_nat(0u);
v___x_1084_ = lean_array_get_size(v_buckets_1081_);
v___x_1085_ = lean_nat_dec_lt(v___x_1083_, v___x_1084_);
if (v___x_1085_ == 0)
{
lean_object* v___x_1086_; 
lean_dec_ref(v_buckets_1081_);
v___x_1086_ = lp_aesop_Aesop_Frontend_getGlobalRuleSets(v___x_1082_, v_a_1075_, v_a_1076_);
return v___x_1086_;
}
else
{
uint8_t v___x_1087_; 
v___x_1087_ = lean_nat_dec_le(v___x_1084_, v___x_1084_);
if (v___x_1087_ == 0)
{
if (v___x_1085_ == 0)
{
lean_object* v___x_1088_; 
lean_dec_ref(v_buckets_1081_);
v___x_1088_ = lp_aesop_Aesop_Frontend_getGlobalRuleSets(v___x_1082_, v_a_1075_, v_a_1076_);
return v___x_1088_;
}
else
{
size_t v___x_1089_; size_t v___x_1090_; lean_object* v___x_1091_; lean_object* v___x_1092_; 
v___x_1089_ = ((size_t)0ULL);
v___x_1090_ = lean_usize_of_nat(v___x_1084_);
v___x_1091_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Frontend_getDefaultGlobalRuleSets_spec__1(v_buckets_1081_, v___x_1089_, v___x_1090_, v___x_1082_);
lean_dec_ref(v_buckets_1081_);
v___x_1092_ = lp_aesop_Aesop_Frontend_getGlobalRuleSets(v___x_1091_, v_a_1075_, v_a_1076_);
return v___x_1092_;
}
}
else
{
size_t v___x_1093_; size_t v___x_1094_; lean_object* v___x_1095_; lean_object* v___x_1096_; 
v___x_1093_ = ((size_t)0ULL);
v___x_1094_ = lean_usize_of_nat(v___x_1084_);
v___x_1095_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Frontend_getDefaultGlobalRuleSets_spec__1(v_buckets_1081_, v___x_1093_, v___x_1094_, v___x_1082_);
lean_dec_ref(v_buckets_1081_);
v___x_1096_ = lp_aesop_Aesop_Frontend_getGlobalRuleSets(v___x_1095_, v_a_1075_, v_a_1076_);
return v___x_1096_;
}
}
}
else
{
lean_object* v_a_1097_; lean_object* v___x_1099_; uint8_t v_isShared_1100_; uint8_t v_isSharedCheck_1109_; 
v_a_1097_ = lean_ctor_get(v___x_1078_, 0);
v_isSharedCheck_1109_ = !lean_is_exclusive(v___x_1078_);
if (v_isSharedCheck_1109_ == 0)
{
v___x_1099_ = v___x_1078_;
v_isShared_1100_ = v_isSharedCheck_1109_;
goto v_resetjp_1098_;
}
else
{
lean_inc(v_a_1097_);
lean_dec(v___x_1078_);
v___x_1099_ = lean_box(0);
v_isShared_1100_ = v_isSharedCheck_1109_;
goto v_resetjp_1098_;
}
v_resetjp_1098_:
{
lean_object* v_ref_1101_; lean_object* v___x_1102_; lean_object* v___x_1103_; lean_object* v___x_1104_; lean_object* v___x_1105_; lean_object* v___x_1107_; 
v_ref_1101_ = lean_ctor_get(v_a_1075_, 5);
v___x_1102_ = lean_io_error_to_string(v_a_1097_);
v___x_1103_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_1103_, 0, v___x_1102_);
v___x_1104_ = l_Lean_MessageData_ofFormat(v___x_1103_);
lean_inc(v_ref_1101_);
v___x_1105_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1105_, 0, v_ref_1101_);
lean_ctor_set(v___x_1105_, 1, v___x_1104_);
if (v_isShared_1100_ == 0)
{
lean_ctor_set(v___x_1099_, 0, v___x_1105_);
v___x_1107_ = v___x_1099_;
goto v_reusejp_1106_;
}
else
{
lean_object* v_reuseFailAlloc_1108_; 
v_reuseFailAlloc_1108_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1108_, 0, v___x_1105_);
v___x_1107_ = v_reuseFailAlloc_1108_;
goto v_reusejp_1106_;
}
v_reusejp_1106_:
{
return v___x_1107_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_getDefaultGlobalRuleSets___boxed(lean_object* v_a_1110_, lean_object* v_a_1111_, lean_object* v_a_1112_){
_start:
{
lean_object* v_res_1113_; 
v_res_1113_ = lp_aesop_Aesop_Frontend_getDefaultGlobalRuleSets(v_a_1110_, v_a_1111_);
lean_dec(v_a_1111_);
lean_dec_ref(v_a_1110_);
return v_res_1113_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00Aesop_Frontend_getDeclaredGlobalRuleSets_spec__0(lean_object* v_x_1114_, lean_object* v_x_1115_){
_start:
{
if (lean_obj_tag(v_x_1115_) == 0)
{
return v_x_1114_;
}
else
{
lean_object* v_key_1116_; lean_object* v_value_1117_; lean_object* v_tail_1118_; lean_object* v___x_1119_; lean_object* v___x_1120_; 
v_key_1116_ = lean_ctor_get(v_x_1115_, 0);
v_value_1117_ = lean_ctor_get(v_x_1115_, 1);
v_tail_1118_ = lean_ctor_get(v_x_1115_, 2);
lean_inc(v_value_1117_);
lean_inc(v_key_1116_);
v___x_1119_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1119_, 0, v_key_1116_);
lean_ctor_set(v___x_1119_, 1, v_value_1117_);
v___x_1120_ = lean_array_push(v_x_1114_, v___x_1119_);
v_x_1114_ = v___x_1120_;
v_x_1115_ = v_tail_1118_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00Aesop_Frontend_getDeclaredGlobalRuleSets_spec__0___boxed(lean_object* v_x_1122_, lean_object* v_x_1123_){
_start:
{
lean_object* v_res_1124_; 
v_res_1124_ = lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00Aesop_Frontend_getDeclaredGlobalRuleSets_spec__0(v_x_1122_, v_x_1123_);
lean_dec(v_x_1123_);
return v_res_1124_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Frontend_getDeclaredGlobalRuleSets_spec__2(lean_object* v_as_1125_, size_t v_i_1126_, size_t v_stop_1127_, lean_object* v_b_1128_){
_start:
{
uint8_t v___x_1129_; 
v___x_1129_ = lean_usize_dec_eq(v_i_1126_, v_stop_1127_);
if (v___x_1129_ == 0)
{
lean_object* v___x_1130_; lean_object* v___x_1131_; size_t v___x_1132_; size_t v___x_1133_; 
v___x_1130_ = lean_array_uget_borrowed(v_as_1125_, v_i_1126_);
v___x_1131_ = lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00Aesop_Frontend_getDeclaredGlobalRuleSets_spec__0(v_b_1128_, v___x_1130_);
v___x_1132_ = ((size_t)1ULL);
v___x_1133_ = lean_usize_add(v_i_1126_, v___x_1132_);
v_i_1126_ = v___x_1133_;
v_b_1128_ = v___x_1131_;
goto _start;
}
else
{
return v_b_1128_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Frontend_getDeclaredGlobalRuleSets_spec__2___boxed(lean_object* v_as_1135_, lean_object* v_i_1136_, lean_object* v_stop_1137_, lean_object* v_b_1138_){
_start:
{
size_t v_i_boxed_1139_; size_t v_stop_boxed_1140_; lean_object* v_res_1141_; 
v_i_boxed_1139_ = lean_unbox_usize(v_i_1136_);
lean_dec(v_i_1136_);
v_stop_boxed_1140_ = lean_unbox_usize(v_stop_1137_);
lean_dec(v_stop_1137_);
v_res_1141_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Frontend_getDeclaredGlobalRuleSets_spec__2(v_as_1135_, v_i_boxed_1139_, v_stop_boxed_1140_, v_b_1138_);
lean_dec_ref(v_as_1135_);
return v_res_1141_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Frontend_getDeclaredGlobalRuleSets_spec__1(size_t v_sz_1142_, size_t v_i_1143_, lean_object* v_bs_1144_, lean_object* v___y_1145_, lean_object* v___y_1146_){
_start:
{
uint8_t v___x_1148_; 
v___x_1148_ = lean_usize_dec_lt(v_i_1143_, v_sz_1142_);
if (v___x_1148_ == 0)
{
lean_object* v___x_1149_; 
v___x_1149_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1149_, 0, v_bs_1144_);
return v___x_1149_;
}
else
{
lean_object* v_v_1150_; lean_object* v_fst_1151_; lean_object* v___x_1153_; uint8_t v_isShared_1154_; uint8_t v_isSharedCheck_1174_; 
v_v_1150_ = lean_array_uget(v_bs_1144_, v_i_1143_);
v_fst_1151_ = lean_ctor_get(v_v_1150_, 0);
v_isSharedCheck_1174_ = !lean_is_exclusive(v_v_1150_);
if (v_isSharedCheck_1174_ == 0)
{
lean_object* v_unused_1175_; 
v_unused_1175_ = lean_ctor_get(v_v_1150_, 1);
lean_dec(v_unused_1175_);
v___x_1153_ = v_v_1150_;
v_isShared_1154_ = v_isSharedCheck_1174_;
goto v_resetjp_1152_;
}
else
{
lean_inc(v_fst_1151_);
lean_dec(v_v_1150_);
v___x_1153_ = lean_box(0);
v_isShared_1154_ = v_isSharedCheck_1174_;
goto v_resetjp_1152_;
}
v_resetjp_1152_:
{
lean_object* v___x_1155_; 
lean_inc(v_fst_1151_);
v___x_1155_ = lp_aesop_Aesop_Frontend_getGlobalRuleSet(v_fst_1151_, v___y_1145_, v___y_1146_);
if (lean_obj_tag(v___x_1155_) == 0)
{
lean_object* v_a_1156_; lean_object* v___x_1157_; lean_object* v_bs_x27_1158_; lean_object* v___x_1160_; 
v_a_1156_ = lean_ctor_get(v___x_1155_, 0);
lean_inc(v_a_1156_);
lean_dec_ref_known(v___x_1155_, 1);
v___x_1157_ = lean_unsigned_to_nat(0u);
v_bs_x27_1158_ = lean_array_uset(v_bs_1144_, v_i_1143_, v___x_1157_);
if (v_isShared_1154_ == 0)
{
lean_ctor_set(v___x_1153_, 1, v_a_1156_);
v___x_1160_ = v___x_1153_;
goto v_reusejp_1159_;
}
else
{
lean_object* v_reuseFailAlloc_1165_; 
v_reuseFailAlloc_1165_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1165_, 0, v_fst_1151_);
lean_ctor_set(v_reuseFailAlloc_1165_, 1, v_a_1156_);
v___x_1160_ = v_reuseFailAlloc_1165_;
goto v_reusejp_1159_;
}
v_reusejp_1159_:
{
size_t v___x_1161_; size_t v___x_1162_; lean_object* v___x_1163_; 
v___x_1161_ = ((size_t)1ULL);
v___x_1162_ = lean_usize_add(v_i_1143_, v___x_1161_);
v___x_1163_ = lean_array_uset(v_bs_x27_1158_, v_i_1143_, v___x_1160_);
v_i_1143_ = v___x_1162_;
v_bs_1144_ = v___x_1163_;
goto _start;
}
}
else
{
lean_object* v_a_1166_; lean_object* v___x_1168_; uint8_t v_isShared_1169_; uint8_t v_isSharedCheck_1173_; 
lean_del_object(v___x_1153_);
lean_dec(v_fst_1151_);
lean_dec_ref(v_bs_1144_);
v_a_1166_ = lean_ctor_get(v___x_1155_, 0);
v_isSharedCheck_1173_ = !lean_is_exclusive(v___x_1155_);
if (v_isSharedCheck_1173_ == 0)
{
v___x_1168_ = v___x_1155_;
v_isShared_1169_ = v_isSharedCheck_1173_;
goto v_resetjp_1167_;
}
else
{
lean_inc(v_a_1166_);
lean_dec(v___x_1155_);
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
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Frontend_getDeclaredGlobalRuleSets_spec__1___boxed(lean_object* v_sz_1176_, lean_object* v_i_1177_, lean_object* v_bs_1178_, lean_object* v___y_1179_, lean_object* v___y_1180_, lean_object* v___y_1181_){
_start:
{
size_t v_sz_boxed_1182_; size_t v_i_boxed_1183_; lean_object* v_res_1184_; 
v_sz_boxed_1182_ = lean_unbox_usize(v_sz_1176_);
lean_dec(v_sz_1176_);
v_i_boxed_1183_ = lean_unbox_usize(v_i_1177_);
lean_dec(v_i_1177_);
v_res_1184_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Frontend_getDeclaredGlobalRuleSets_spec__1(v_sz_boxed_1182_, v_i_boxed_1183_, v_bs_1178_, v___y_1179_, v___y_1180_);
lean_dec(v___y_1180_);
lean_dec_ref(v___y_1179_);
return v_res_1184_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_getDeclaredGlobalRuleSets(lean_object* v_a_1185_, lean_object* v_a_1186_){
_start:
{
lean_object* v___x_1188_; 
v___x_1188_ = lp_aesop_Aesop_getDeclaredRuleSets();
if (lean_obj_tag(v___x_1188_) == 0)
{
lean_object* v_a_1189_; lean_object* v_size_1190_; lean_object* v_buckets_1191_; lean_object* v___x_1192_; lean_object* v___x_1193_; lean_object* v___x_1194_; uint8_t v___x_1195_; 
v_a_1189_ = lean_ctor_get(v___x_1188_, 0);
lean_inc(v_a_1189_);
lean_dec_ref_known(v___x_1188_, 1);
v_size_1190_ = lean_ctor_get(v_a_1189_, 0);
lean_inc(v_size_1190_);
v_buckets_1191_ = lean_ctor_get(v_a_1189_, 1);
lean_inc_ref(v_buckets_1191_);
lean_dec(v_a_1189_);
v___x_1192_ = lean_mk_empty_array_with_capacity(v_size_1190_);
lean_dec(v_size_1190_);
v___x_1193_ = lean_unsigned_to_nat(0u);
v___x_1194_ = lean_array_get_size(v_buckets_1191_);
v___x_1195_ = lean_nat_dec_lt(v___x_1193_, v___x_1194_);
if (v___x_1195_ == 0)
{
size_t v_sz_1196_; size_t v___x_1197_; lean_object* v___x_1198_; 
lean_dec_ref(v_buckets_1191_);
v_sz_1196_ = lean_array_size(v___x_1192_);
v___x_1197_ = ((size_t)0ULL);
v___x_1198_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Frontend_getDeclaredGlobalRuleSets_spec__1(v_sz_1196_, v___x_1197_, v___x_1192_, v_a_1185_, v_a_1186_);
return v___x_1198_;
}
else
{
uint8_t v___x_1199_; 
v___x_1199_ = lean_nat_dec_le(v___x_1194_, v___x_1194_);
if (v___x_1199_ == 0)
{
if (v___x_1195_ == 0)
{
size_t v_sz_1200_; size_t v___x_1201_; lean_object* v___x_1202_; 
lean_dec_ref(v_buckets_1191_);
v_sz_1200_ = lean_array_size(v___x_1192_);
v___x_1201_ = ((size_t)0ULL);
v___x_1202_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Frontend_getDeclaredGlobalRuleSets_spec__1(v_sz_1200_, v___x_1201_, v___x_1192_, v_a_1185_, v_a_1186_);
return v___x_1202_;
}
else
{
size_t v___x_1203_; size_t v___x_1204_; lean_object* v___x_1205_; size_t v_sz_1206_; lean_object* v___x_1207_; 
v___x_1203_ = ((size_t)0ULL);
v___x_1204_ = lean_usize_of_nat(v___x_1194_);
v___x_1205_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Frontend_getDeclaredGlobalRuleSets_spec__2(v_buckets_1191_, v___x_1203_, v___x_1204_, v___x_1192_);
lean_dec_ref(v_buckets_1191_);
v_sz_1206_ = lean_array_size(v___x_1205_);
v___x_1207_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Frontend_getDeclaredGlobalRuleSets_spec__1(v_sz_1206_, v___x_1203_, v___x_1205_, v_a_1185_, v_a_1186_);
return v___x_1207_;
}
}
else
{
size_t v___x_1208_; size_t v___x_1209_; lean_object* v___x_1210_; size_t v_sz_1211_; lean_object* v___x_1212_; 
v___x_1208_ = ((size_t)0ULL);
v___x_1209_ = lean_usize_of_nat(v___x_1194_);
v___x_1210_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Frontend_getDeclaredGlobalRuleSets_spec__2(v_buckets_1191_, v___x_1208_, v___x_1209_, v___x_1192_);
lean_dec_ref(v_buckets_1191_);
v_sz_1211_ = lean_array_size(v___x_1210_);
v___x_1212_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Frontend_getDeclaredGlobalRuleSets_spec__1(v_sz_1211_, v___x_1208_, v___x_1210_, v_a_1185_, v_a_1186_);
return v___x_1212_;
}
}
}
else
{
lean_object* v_a_1213_; lean_object* v___x_1215_; uint8_t v_isShared_1216_; uint8_t v_isSharedCheck_1225_; 
v_a_1213_ = lean_ctor_get(v___x_1188_, 0);
v_isSharedCheck_1225_ = !lean_is_exclusive(v___x_1188_);
if (v_isSharedCheck_1225_ == 0)
{
v___x_1215_ = v___x_1188_;
v_isShared_1216_ = v_isSharedCheck_1225_;
goto v_resetjp_1214_;
}
else
{
lean_inc(v_a_1213_);
lean_dec(v___x_1188_);
v___x_1215_ = lean_box(0);
v_isShared_1216_ = v_isSharedCheck_1225_;
goto v_resetjp_1214_;
}
v_resetjp_1214_:
{
lean_object* v_ref_1217_; lean_object* v___x_1218_; lean_object* v___x_1219_; lean_object* v___x_1220_; lean_object* v___x_1221_; lean_object* v___x_1223_; 
v_ref_1217_ = lean_ctor_get(v_a_1185_, 5);
v___x_1218_ = lean_io_error_to_string(v_a_1213_);
v___x_1219_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_1219_, 0, v___x_1218_);
v___x_1220_ = l_Lean_MessageData_ofFormat(v___x_1219_);
lean_inc(v_ref_1217_);
v___x_1221_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1221_, 0, v_ref_1217_);
lean_ctor_set(v___x_1221_, 1, v___x_1220_);
if (v_isShared_1216_ == 0)
{
lean_ctor_set(v___x_1215_, 0, v___x_1221_);
v___x_1223_ = v___x_1215_;
goto v_reusejp_1222_;
}
else
{
lean_object* v_reuseFailAlloc_1224_; 
v_reuseFailAlloc_1224_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1224_, 0, v___x_1221_);
v___x_1223_ = v_reuseFailAlloc_1224_;
goto v_reusejp_1222_;
}
v_reusejp_1222_:
{
return v___x_1223_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_getDeclaredGlobalRuleSets___boxed(lean_object* v_a_1226_, lean_object* v_a_1227_, lean_object* v_a_1228_){
_start:
{
lean_object* v_res_1229_; 
v_res_1229_ = lp_aesop_Aesop_Frontend_getDeclaredGlobalRuleSets(v_a_1226_, v_a_1227_);
lean_dec(v_a_1227_);
lean_dec_ref(v_a_1226_);
return v_res_1229_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_modifyGetGlobalRuleSet___redArg___lam__0(lean_object* v_x_1230_){
_start:
{
lean_object* v___x_1231_; 
v___x_1231_ = l_Lean_Meta_Simp_instInhabitedSimprocs_default;
return v___x_1231_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_modifyGetGlobalRuleSet___redArg___lam__0___boxed(lean_object* v_x_1232_){
_start:
{
lean_object* v_res_1233_; 
v_res_1233_ = lp_aesop_Aesop_Frontend_modifyGetGlobalRuleSet___redArg___lam__0(v_x_1232_);
lean_dec_ref(v_x_1232_);
return v_res_1233_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_modifyGetGlobalRuleSet___redArg___lam__1(lean_object* v_x_1234_){
_start:
{
lean_object* v___x_1235_; 
v___x_1235_ = l_Lean_Meta_instInhabitedSimpTheorems_default;
return v___x_1235_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_modifyGetGlobalRuleSet___redArg___lam__1___boxed(lean_object* v_x_1236_){
_start:
{
lean_object* v_res_1237_; 
v_res_1237_ = lp_aesop_Aesop_Frontend_modifyGetGlobalRuleSet___redArg___lam__1(v_x_1236_);
lean_dec_ref(v_x_1236_);
return v_res_1237_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_modifyGetGlobalRuleSet___redArg___lam__2(lean_object* v_x_1238_){
_start:
{
lean_object* v___x_1239_; 
v___x_1239_ = lp_aesop_Aesop_instInhabitedBaseRuleSet_default;
return v___x_1239_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_modifyGetGlobalRuleSet___redArg___lam__2___boxed(lean_object* v_x_1240_){
_start:
{
lean_object* v_res_1241_; 
v_res_1241_ = lp_aesop_Aesop_Frontend_modifyGetGlobalRuleSet___redArg___lam__2(v_x_1240_);
lean_dec_ref(v_x_1240_);
return v_res_1241_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_modifyGetGlobalRuleSet___redArg___lam__3(lean_object* v_snd_1242_, lean_object* v_x_1243_){
_start:
{
lean_object* v_toBaseRuleSet_1244_; 
v_toBaseRuleSet_1244_ = lean_ctor_get(v_snd_1242_, 0);
lean_inc_ref(v_toBaseRuleSet_1244_);
return v_toBaseRuleSet_1244_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_modifyGetGlobalRuleSet___redArg___lam__3___boxed(lean_object* v_snd_1245_, lean_object* v_x_1246_){
_start:
{
lean_object* v_res_1247_; 
v_res_1247_ = lp_aesop_Aesop_Frontend_modifyGetGlobalRuleSet___redArg___lam__3(v_snd_1245_, v_x_1246_);
lean_dec_ref(v_x_1246_);
lean_dec_ref(v_snd_1245_);
return v_res_1247_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_modifyGetGlobalRuleSet___redArg___lam__4(lean_object* v_snd_1248_, lean_object* v_x_1249_){
_start:
{
lean_object* v_simpTheorems_1250_; 
v_simpTheorems_1250_ = lean_ctor_get(v_snd_1248_, 1);
lean_inc_ref(v_simpTheorems_1250_);
return v_simpTheorems_1250_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_modifyGetGlobalRuleSet___redArg___lam__4___boxed(lean_object* v_snd_1251_, lean_object* v_x_1252_){
_start:
{
lean_object* v_res_1253_; 
v_res_1253_ = lp_aesop_Aesop_Frontend_modifyGetGlobalRuleSet___redArg___lam__4(v_snd_1251_, v_x_1252_);
lean_dec_ref(v_x_1252_);
lean_dec_ref(v_snd_1251_);
return v_res_1253_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_modifyGetGlobalRuleSet___redArg___lam__5(lean_object* v_toPure_1254_, lean_object* v_fst_1255_, lean_object* v_____r_1256_){
_start:
{
lean_object* v___x_1257_; 
v___x_1257_ = lean_apply_2(v_toPure_1254_, lean_box(0), v_fst_1255_);
return v___x_1257_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_modifyGetGlobalRuleSet___redArg___lam__6(lean_object* v_fst_1258_, lean_object* v_fst_1259_, lean_object* v_snd_1260_, lean_object* v___x_1261_, lean_object* v___x_1262_, lean_object* v___x_1263_, lean_object* v_f_1264_, lean_object* v_toPure_1265_, lean_object* v___f_1266_, lean_object* v___f_1267_, lean_object* v___f_1268_, lean_object* v_inst_1269_, lean_object* v_toBind_1270_, lean_object* v_env_1271_){
_start:
{
lean_object* v_ext_1272_; lean_object* v_toEnvExtension_1273_; lean_object* v_ext_1274_; lean_object* v_toEnvExtension_1275_; lean_object* v_ext_1276_; lean_object* v_toEnvExtension_1277_; lean_object* v_asyncMode_1278_; lean_object* v_asyncMode_1279_; lean_object* v_asyncMode_1280_; lean_object* v_base_1281_; lean_object* v_simpTheorems_1282_; lean_object* v_simprocs_1283_; lean_object* v_rs_1284_; lean_object* v___x_1285_; lean_object* v_fst_1286_; lean_object* v_snd_1287_; lean_object* v___f_1288_; lean_object* v___f_1289_; lean_object* v___f_1290_; lean_object* v_env_1291_; lean_object* v_env_1292_; lean_object* v_env_1293_; lean_object* v_env_1294_; lean_object* v_env_1295_; lean_object* v___x_1296_; lean_object* v___x_1297_; 
v_ext_1272_ = lean_ctor_get(v_fst_1258_, 1);
v_toEnvExtension_1273_ = lean_ctor_get(v_ext_1272_, 0);
v_ext_1274_ = lean_ctor_get(v_fst_1259_, 1);
v_toEnvExtension_1275_ = lean_ctor_get(v_ext_1274_, 0);
v_ext_1276_ = lean_ctor_get(v_snd_1260_, 1);
v_toEnvExtension_1277_ = lean_ctor_get(v_ext_1276_, 0);
v_asyncMode_1278_ = lean_ctor_get(v_toEnvExtension_1273_, 2);
v_asyncMode_1279_ = lean_ctor_get(v_toEnvExtension_1275_, 2);
v_asyncMode_1280_ = lean_ctor_get(v_toEnvExtension_1277_, 2);
lean_inc_ref_n(v_env_1271_, 3);
v_base_1281_ = l_Lean_ScopedEnvExtension_getState___redArg(v___x_1261_, v_fst_1258_, v_env_1271_, v_asyncMode_1278_);
v_simpTheorems_1282_ = l_Lean_ScopedEnvExtension_getState___redArg(v___x_1262_, v_fst_1259_, v_env_1271_, v_asyncMode_1279_);
v_simprocs_1283_ = l_Lean_ScopedEnvExtension_getState___redArg(v___x_1263_, v_snd_1260_, v_env_1271_, v_asyncMode_1280_);
v_rs_1284_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_rs_1284_, 0, v_base_1281_);
lean_ctor_set(v_rs_1284_, 1, v_simpTheorems_1282_);
lean_ctor_set(v_rs_1284_, 2, v_simprocs_1283_);
v___x_1285_ = lean_apply_1(v_f_1264_, v_rs_1284_);
v_fst_1286_ = lean_ctor_get(v___x_1285_, 0);
lean_inc(v_fst_1286_);
v_snd_1287_ = lean_ctor_get(v___x_1285_, 1);
lean_inc_n(v_snd_1287_, 2);
lean_dec_ref(v___x_1285_);
v___f_1288_ = lean_alloc_closure((void*)(lp_aesop_Aesop_Frontend_modifyGetGlobalRuleSet___redArg___lam__3___boxed), 2, 1);
lean_closure_set(v___f_1288_, 0, v_snd_1287_);
v___f_1289_ = lean_alloc_closure((void*)(lp_aesop_Aesop_Frontend_modifyGetGlobalRuleSet___redArg___lam__4___boxed), 2, 1);
lean_closure_set(v___f_1289_, 0, v_snd_1287_);
v___f_1290_ = lean_alloc_closure((void*)(lp_aesop_Aesop_Frontend_modifyGetGlobalRuleSet___redArg___lam__5), 3, 2);
lean_closure_set(v___f_1290_, 0, v_toPure_1265_);
lean_closure_set(v___f_1290_, 1, v_fst_1286_);
lean_inc_ref(v_fst_1258_);
v_env_1291_ = l_Lean_ScopedEnvExtension_modifyState___redArg(v_fst_1258_, v_env_1271_, v___f_1266_);
lean_inc_ref(v_fst_1259_);
v_env_1292_ = l_Lean_ScopedEnvExtension_modifyState___redArg(v_fst_1259_, v_env_1291_, v___f_1267_);
v_env_1293_ = l_Lean_ScopedEnvExtension_modifyState___redArg(v_snd_1260_, v_env_1292_, v___f_1268_);
v_env_1294_ = l_Lean_ScopedEnvExtension_modifyState___redArg(v_fst_1258_, v_env_1293_, v___f_1288_);
v_env_1295_ = l_Lean_ScopedEnvExtension_modifyState___redArg(v_fst_1259_, v_env_1294_, v___f_1289_);
v___x_1296_ = l_Lean_setEnv___redArg(v_inst_1269_, v_env_1295_);
v___x_1297_ = lean_apply_4(v_toBind_1270_, lean_box(0), lean_box(0), v___x_1296_, v___f_1290_);
return v___x_1297_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_modifyGetGlobalRuleSet___redArg___lam__6___boxed(lean_object* v_fst_1298_, lean_object* v_fst_1299_, lean_object* v_snd_1300_, lean_object* v___x_1301_, lean_object* v___x_1302_, lean_object* v___x_1303_, lean_object* v_f_1304_, lean_object* v_toPure_1305_, lean_object* v___f_1306_, lean_object* v___f_1307_, lean_object* v___f_1308_, lean_object* v_inst_1309_, lean_object* v_toBind_1310_, lean_object* v_env_1311_){
_start:
{
lean_object* v_res_1312_; 
v_res_1312_ = lp_aesop_Aesop_Frontend_modifyGetGlobalRuleSet___redArg___lam__6(v_fst_1298_, v_fst_1299_, v_snd_1300_, v___x_1301_, v___x_1302_, v___x_1303_, v_f_1304_, v_toPure_1305_, v___f_1306_, v___f_1307_, v___f_1308_, v_inst_1309_, v_toBind_1310_, v_env_1311_);
lean_dec_ref(v___x_1303_);
lean_dec_ref(v___x_1302_);
lean_dec_ref(v___x_1301_);
return v_res_1312_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_modifyGetGlobalRuleSet___redArg___lam__7(lean_object* v_inst_1313_, lean_object* v___x_1314_, lean_object* v___x_1315_, lean_object* v___x_1316_, lean_object* v_f_1317_, lean_object* v_toPure_1318_, lean_object* v___f_1319_, lean_object* v___f_1320_, lean_object* v___f_1321_, lean_object* v_toBind_1322_, lean_object* v_____x_1323_){
_start:
{
lean_object* v_snd_1324_; lean_object* v_snd_1325_; lean_object* v_snd_1326_; lean_object* v_fst_1327_; lean_object* v_fst_1328_; lean_object* v_snd_1329_; lean_object* v_getEnv_1330_; lean_object* v___f_1331_; lean_object* v___x_1332_; 
v_snd_1324_ = lean_ctor_get(v_____x_1323_, 1);
v_snd_1325_ = lean_ctor_get(v_snd_1324_, 1);
lean_inc(v_snd_1325_);
v_snd_1326_ = lean_ctor_get(v_snd_1325_, 1);
lean_inc(v_snd_1326_);
v_fst_1327_ = lean_ctor_get(v_____x_1323_, 0);
lean_inc(v_fst_1327_);
lean_dec_ref(v_____x_1323_);
v_fst_1328_ = lean_ctor_get(v_snd_1325_, 0);
lean_inc(v_fst_1328_);
lean_dec(v_snd_1325_);
v_snd_1329_ = lean_ctor_get(v_snd_1326_, 1);
lean_inc(v_snd_1329_);
lean_dec(v_snd_1326_);
v_getEnv_1330_ = lean_ctor_get(v_inst_1313_, 0);
lean_inc(v_getEnv_1330_);
lean_inc(v_toBind_1322_);
v___f_1331_ = lean_alloc_closure((void*)(lp_aesop_Aesop_Frontend_modifyGetGlobalRuleSet___redArg___lam__6___boxed), 14, 13);
lean_closure_set(v___f_1331_, 0, v_fst_1327_);
lean_closure_set(v___f_1331_, 1, v_fst_1328_);
lean_closure_set(v___f_1331_, 2, v_snd_1329_);
lean_closure_set(v___f_1331_, 3, v___x_1314_);
lean_closure_set(v___f_1331_, 4, v___x_1315_);
lean_closure_set(v___f_1331_, 5, v___x_1316_);
lean_closure_set(v___f_1331_, 6, v_f_1317_);
lean_closure_set(v___f_1331_, 7, v_toPure_1318_);
lean_closure_set(v___f_1331_, 8, v___f_1319_);
lean_closure_set(v___f_1331_, 9, v___f_1320_);
lean_closure_set(v___f_1331_, 10, v___f_1321_);
lean_closure_set(v___f_1331_, 11, v_inst_1313_);
lean_closure_set(v___f_1331_, 12, v_toBind_1322_);
v___x_1332_ = lean_apply_4(v_toBind_1322_, lean_box(0), lean_box(0), v_getEnv_1330_, v___f_1331_);
return v___x_1332_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_modifyGetGlobalRuleSet___redArg(lean_object* v_inst_1336_, lean_object* v_inst_1337_, lean_object* v_inst_1338_, lean_object* v_inst_1339_, lean_object* v_inst_1340_, lean_object* v_rsName_1341_, lean_object* v_f_1342_){
_start:
{
lean_object* v_toApplicative_1343_; lean_object* v_toBind_1344_; lean_object* v_toPure_1345_; lean_object* v___f_1346_; lean_object* v___f_1347_; lean_object* v___f_1348_; lean_object* v___x_1349_; lean_object* v___x_1350_; lean_object* v___x_1351_; lean_object* v___x_1352_; lean_object* v___f_1353_; lean_object* v___x_1354_; 
v_toApplicative_1343_ = lean_ctor_get(v_inst_1336_, 0);
v_toBind_1344_ = lean_ctor_get(v_inst_1336_, 1);
lean_inc_n(v_toBind_1344_, 2);
v_toPure_1345_ = lean_ctor_get(v_toApplicative_1343_, 1);
lean_inc(v_toPure_1345_);
v___f_1346_ = ((lean_object*)(lp_aesop_Aesop_Frontend_modifyGetGlobalRuleSet___redArg___closed__0));
v___f_1347_ = ((lean_object*)(lp_aesop_Aesop_Frontend_modifyGetGlobalRuleSet___redArg___closed__1));
v___f_1348_ = ((lean_object*)(lp_aesop_Aesop_Frontend_modifyGetGlobalRuleSet___redArg___closed__2));
v___x_1349_ = lp_aesop_Aesop_instInhabitedBaseRuleSet_default;
v___x_1350_ = l_Lean_Meta_instInhabitedSimpTheorems_default;
v___x_1351_ = l_Lean_Meta_Simp_instInhabitedSimprocs_default;
v___x_1352_ = lp_aesop_Aesop_Frontend_getGlobalRuleSetData___redArg(v_inst_1336_, v_inst_1337_, v_inst_1338_, v_inst_1339_, v_rsName_1341_);
v___f_1353_ = lean_alloc_closure((void*)(lp_aesop_Aesop_Frontend_modifyGetGlobalRuleSet___redArg___lam__7), 11, 10);
lean_closure_set(v___f_1353_, 0, v_inst_1340_);
lean_closure_set(v___f_1353_, 1, v___x_1349_);
lean_closure_set(v___f_1353_, 2, v___x_1350_);
lean_closure_set(v___f_1353_, 3, v___x_1351_);
lean_closure_set(v___f_1353_, 4, v_f_1342_);
lean_closure_set(v___f_1353_, 5, v_toPure_1345_);
lean_closure_set(v___f_1353_, 6, v___f_1348_);
lean_closure_set(v___f_1353_, 7, v___f_1347_);
lean_closure_set(v___f_1353_, 8, v___f_1346_);
lean_closure_set(v___f_1353_, 9, v_toBind_1344_);
v___x_1354_ = lean_apply_4(v_toBind_1344_, lean_box(0), lean_box(0), v___x_1352_, v___f_1353_);
return v___x_1354_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_modifyGetGlobalRuleSet(lean_object* v_m_1355_, lean_object* v_inst_1356_, lean_object* v_inst_1357_, lean_object* v_inst_1358_, lean_object* v_inst_1359_, lean_object* v_inst_1360_, lean_object* v_00_u03b1_1361_, lean_object* v_rsName_1362_, lean_object* v_f_1363_){
_start:
{
lean_object* v___x_1364_; 
v___x_1364_ = lp_aesop_Aesop_Frontend_modifyGetGlobalRuleSet___redArg(v_inst_1356_, v_inst_1357_, v_inst_1358_, v_inst_1359_, v_inst_1360_, v_rsName_1362_, v_f_1363_);
return v___x_1364_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_modifyGlobalRuleSet___lam__0(lean_object* v_f_1365_, lean_object* v_rs_1366_){
_start:
{
lean_object* v___x_1367_; lean_object* v___x_1368_; lean_object* v___x_1369_; 
v___x_1367_ = lean_box(0);
v___x_1368_ = lean_apply_1(v_f_1365_, v_rs_1366_);
v___x_1369_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1369_, 0, v___x_1367_);
lean_ctor_set(v___x_1369_, 1, v___x_1368_);
return v___x_1369_;
}
}
static lean_object* _init_lp_aesop_Lean_setEnv___at___00Aesop_Frontend_modifyGetGlobalRuleSet___at___00Aesop_Frontend_modifyGlobalRuleSet_spec__0_spec__0___redArg___closed__0(void){
_start:
{
lean_object* v___x_1370_; 
v___x_1370_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_1370_;
}
}
static lean_object* _init_lp_aesop_Lean_setEnv___at___00Aesop_Frontend_modifyGetGlobalRuleSet___at___00Aesop_Frontend_modifyGlobalRuleSet_spec__0_spec__0___redArg___closed__1(void){
_start:
{
lean_object* v___x_1371_; lean_object* v___x_1372_; 
v___x_1371_ = lean_obj_once(&lp_aesop_Lean_setEnv___at___00Aesop_Frontend_modifyGetGlobalRuleSet___at___00Aesop_Frontend_modifyGlobalRuleSet_spec__0_spec__0___redArg___closed__0, &lp_aesop_Lean_setEnv___at___00Aesop_Frontend_modifyGetGlobalRuleSet___at___00Aesop_Frontend_modifyGlobalRuleSet_spec__0_spec__0___redArg___closed__0_once, _init_lp_aesop_Lean_setEnv___at___00Aesop_Frontend_modifyGetGlobalRuleSet___at___00Aesop_Frontend_modifyGlobalRuleSet_spec__0_spec__0___redArg___closed__0);
v___x_1372_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1372_, 0, v___x_1371_);
return v___x_1372_;
}
}
static lean_object* _init_lp_aesop_Lean_setEnv___at___00Aesop_Frontend_modifyGetGlobalRuleSet___at___00Aesop_Frontend_modifyGlobalRuleSet_spec__0_spec__0___redArg___closed__2(void){
_start:
{
lean_object* v___x_1373_; lean_object* v___x_1374_; 
v___x_1373_ = lean_obj_once(&lp_aesop_Lean_setEnv___at___00Aesop_Frontend_modifyGetGlobalRuleSet___at___00Aesop_Frontend_modifyGlobalRuleSet_spec__0_spec__0___redArg___closed__1, &lp_aesop_Lean_setEnv___at___00Aesop_Frontend_modifyGetGlobalRuleSet___at___00Aesop_Frontend_modifyGlobalRuleSet_spec__0_spec__0___redArg___closed__1_once, _init_lp_aesop_Lean_setEnv___at___00Aesop_Frontend_modifyGetGlobalRuleSet___at___00Aesop_Frontend_modifyGlobalRuleSet_spec__0_spec__0___redArg___closed__1);
v___x_1374_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1374_, 0, v___x_1373_);
lean_ctor_set(v___x_1374_, 1, v___x_1373_);
return v___x_1374_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_setEnv___at___00Aesop_Frontend_modifyGetGlobalRuleSet___at___00Aesop_Frontend_modifyGlobalRuleSet_spec__0_spec__0___redArg(lean_object* v_env_1375_, lean_object* v___y_1376_){
_start:
{
lean_object* v___x_1378_; lean_object* v_nextMacroScope_1379_; lean_object* v_ngen_1380_; lean_object* v_auxDeclNGen_1381_; lean_object* v_traceState_1382_; lean_object* v_messages_1383_; lean_object* v_infoState_1384_; lean_object* v_snapshotTasks_1385_; lean_object* v___x_1387_; uint8_t v_isShared_1388_; uint8_t v_isSharedCheck_1396_; 
v___x_1378_ = lean_st_ref_take(v___y_1376_);
v_nextMacroScope_1379_ = lean_ctor_get(v___x_1378_, 1);
v_ngen_1380_ = lean_ctor_get(v___x_1378_, 2);
v_auxDeclNGen_1381_ = lean_ctor_get(v___x_1378_, 3);
v_traceState_1382_ = lean_ctor_get(v___x_1378_, 4);
v_messages_1383_ = lean_ctor_get(v___x_1378_, 6);
v_infoState_1384_ = lean_ctor_get(v___x_1378_, 7);
v_snapshotTasks_1385_ = lean_ctor_get(v___x_1378_, 8);
v_isSharedCheck_1396_ = !lean_is_exclusive(v___x_1378_);
if (v_isSharedCheck_1396_ == 0)
{
lean_object* v_unused_1397_; lean_object* v_unused_1398_; 
v_unused_1397_ = lean_ctor_get(v___x_1378_, 5);
lean_dec(v_unused_1397_);
v_unused_1398_ = lean_ctor_get(v___x_1378_, 0);
lean_dec(v_unused_1398_);
v___x_1387_ = v___x_1378_;
v_isShared_1388_ = v_isSharedCheck_1396_;
goto v_resetjp_1386_;
}
else
{
lean_inc(v_snapshotTasks_1385_);
lean_inc(v_infoState_1384_);
lean_inc(v_messages_1383_);
lean_inc(v_traceState_1382_);
lean_inc(v_auxDeclNGen_1381_);
lean_inc(v_ngen_1380_);
lean_inc(v_nextMacroScope_1379_);
lean_dec(v___x_1378_);
v___x_1387_ = lean_box(0);
v_isShared_1388_ = v_isSharedCheck_1396_;
goto v_resetjp_1386_;
}
v_resetjp_1386_:
{
lean_object* v___x_1389_; lean_object* v___x_1391_; 
v___x_1389_ = lean_obj_once(&lp_aesop_Lean_setEnv___at___00Aesop_Frontend_modifyGetGlobalRuleSet___at___00Aesop_Frontend_modifyGlobalRuleSet_spec__0_spec__0___redArg___closed__2, &lp_aesop_Lean_setEnv___at___00Aesop_Frontend_modifyGetGlobalRuleSet___at___00Aesop_Frontend_modifyGlobalRuleSet_spec__0_spec__0___redArg___closed__2_once, _init_lp_aesop_Lean_setEnv___at___00Aesop_Frontend_modifyGetGlobalRuleSet___at___00Aesop_Frontend_modifyGlobalRuleSet_spec__0_spec__0___redArg___closed__2);
if (v_isShared_1388_ == 0)
{
lean_ctor_set(v___x_1387_, 5, v___x_1389_);
lean_ctor_set(v___x_1387_, 0, v_env_1375_);
v___x_1391_ = v___x_1387_;
goto v_reusejp_1390_;
}
else
{
lean_object* v_reuseFailAlloc_1395_; 
v_reuseFailAlloc_1395_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_1395_, 0, v_env_1375_);
lean_ctor_set(v_reuseFailAlloc_1395_, 1, v_nextMacroScope_1379_);
lean_ctor_set(v_reuseFailAlloc_1395_, 2, v_ngen_1380_);
lean_ctor_set(v_reuseFailAlloc_1395_, 3, v_auxDeclNGen_1381_);
lean_ctor_set(v_reuseFailAlloc_1395_, 4, v_traceState_1382_);
lean_ctor_set(v_reuseFailAlloc_1395_, 5, v___x_1389_);
lean_ctor_set(v_reuseFailAlloc_1395_, 6, v_messages_1383_);
lean_ctor_set(v_reuseFailAlloc_1395_, 7, v_infoState_1384_);
lean_ctor_set(v_reuseFailAlloc_1395_, 8, v_snapshotTasks_1385_);
v___x_1391_ = v_reuseFailAlloc_1395_;
goto v_reusejp_1390_;
}
v_reusejp_1390_:
{
lean_object* v___x_1392_; lean_object* v___x_1393_; lean_object* v___x_1394_; 
v___x_1392_ = lean_st_ref_set(v___y_1376_, v___x_1391_);
v___x_1393_ = lean_box(0);
v___x_1394_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1394_, 0, v___x_1393_);
return v___x_1394_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_setEnv___at___00Aesop_Frontend_modifyGetGlobalRuleSet___at___00Aesop_Frontend_modifyGlobalRuleSet_spec__0_spec__0___redArg___boxed(lean_object* v_env_1399_, lean_object* v___y_1400_, lean_object* v___y_1401_){
_start:
{
lean_object* v_res_1402_; 
v_res_1402_ = lp_aesop_Lean_setEnv___at___00Aesop_Frontend_modifyGetGlobalRuleSet___at___00Aesop_Frontend_modifyGlobalRuleSet_spec__0_spec__0___redArg(v_env_1399_, v___y_1400_);
lean_dec(v___y_1400_);
return v_res_1402_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_modifyGetGlobalRuleSet___at___00Aesop_Frontend_modifyGlobalRuleSet_spec__0___redArg(lean_object* v_rsName_1403_, lean_object* v_f_1404_, lean_object* v___y_1405_, lean_object* v___y_1406_){
_start:
{
lean_object* v___x_1408_; 
v___x_1408_ = lp_aesop_Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_getGlobalRuleSet_spec__0(v_rsName_1403_, v___y_1405_, v___y_1406_);
if (lean_obj_tag(v___x_1408_) == 0)
{
lean_object* v_a_1409_; lean_object* v_snd_1410_; lean_object* v_snd_1411_; lean_object* v_snd_1412_; lean_object* v_fst_1413_; lean_object* v_fst_1414_; lean_object* v_snd_1415_; lean_object* v___x_1416_; lean_object* v_ext_1417_; lean_object* v_toEnvExtension_1418_; lean_object* v_ext_1419_; lean_object* v_toEnvExtension_1420_; lean_object* v_ext_1421_; lean_object* v_toEnvExtension_1422_; lean_object* v_env_1423_; lean_object* v_asyncMode_1424_; lean_object* v_asyncMode_1425_; lean_object* v_asyncMode_1426_; lean_object* v___x_1427_; lean_object* v_base_1428_; lean_object* v___x_1429_; lean_object* v___x_1430_; lean_object* v_simpTheorems_1431_; lean_object* v_simprocs_1432_; lean_object* v_rs_1433_; lean_object* v___x_1434_; lean_object* v_fst_1435_; lean_object* v_snd_1436_; lean_object* v___f_1437_; lean_object* v___f_1438_; lean_object* v___f_1439_; lean_object* v___f_1440_; lean_object* v___f_1441_; lean_object* v_env_1442_; lean_object* v_env_1443_; lean_object* v_env_1444_; lean_object* v_env_1445_; lean_object* v_env_1446_; lean_object* v___x_1447_; lean_object* v___x_1449_; uint8_t v_isShared_1450_; uint8_t v_isSharedCheck_1454_; 
v_a_1409_ = lean_ctor_get(v___x_1408_, 0);
lean_inc(v_a_1409_);
lean_dec_ref_known(v___x_1408_, 1);
v_snd_1410_ = lean_ctor_get(v_a_1409_, 1);
v_snd_1411_ = lean_ctor_get(v_snd_1410_, 1);
lean_inc(v_snd_1411_);
v_snd_1412_ = lean_ctor_get(v_snd_1411_, 1);
lean_inc(v_snd_1412_);
v_fst_1413_ = lean_ctor_get(v_a_1409_, 0);
lean_inc_n(v_fst_1413_, 2);
lean_dec(v_a_1409_);
v_fst_1414_ = lean_ctor_get(v_snd_1411_, 0);
lean_inc_n(v_fst_1414_, 2);
lean_dec(v_snd_1411_);
v_snd_1415_ = lean_ctor_get(v_snd_1412_, 1);
lean_inc(v_snd_1415_);
lean_dec(v_snd_1412_);
v___x_1416_ = lean_st_ref_get(v___y_1406_);
v_ext_1417_ = lean_ctor_get(v_fst_1413_, 1);
v_toEnvExtension_1418_ = lean_ctor_get(v_ext_1417_, 0);
v_ext_1419_ = lean_ctor_get(v_fst_1414_, 1);
v_toEnvExtension_1420_ = lean_ctor_get(v_ext_1419_, 0);
v_ext_1421_ = lean_ctor_get(v_snd_1415_, 1);
v_toEnvExtension_1422_ = lean_ctor_get(v_ext_1421_, 0);
v_env_1423_ = lean_ctor_get(v___x_1416_, 0);
lean_inc_ref_n(v_env_1423_, 4);
lean_dec(v___x_1416_);
v_asyncMode_1424_ = lean_ctor_get(v_toEnvExtension_1418_, 2);
v_asyncMode_1425_ = lean_ctor_get(v_toEnvExtension_1420_, 2);
v_asyncMode_1426_ = lean_ctor_get(v_toEnvExtension_1422_, 2);
v___x_1427_ = lp_aesop_Aesop_instInhabitedBaseRuleSet_default;
v_base_1428_ = l_Lean_ScopedEnvExtension_getState___redArg(v___x_1427_, v_fst_1413_, v_env_1423_, v_asyncMode_1424_);
v___x_1429_ = l_Lean_Meta_instInhabitedSimpTheorems_default;
v___x_1430_ = l_Lean_Meta_Simp_instInhabitedSimprocs_default;
v_simpTheorems_1431_ = l_Lean_ScopedEnvExtension_getState___redArg(v___x_1429_, v_fst_1414_, v_env_1423_, v_asyncMode_1425_);
v_simprocs_1432_ = l_Lean_ScopedEnvExtension_getState___redArg(v___x_1430_, v_snd_1415_, v_env_1423_, v_asyncMode_1426_);
v_rs_1433_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_rs_1433_, 0, v_base_1428_);
lean_ctor_set(v_rs_1433_, 1, v_simpTheorems_1431_);
lean_ctor_set(v_rs_1433_, 2, v_simprocs_1432_);
v___x_1434_ = lean_apply_1(v_f_1404_, v_rs_1433_);
v_fst_1435_ = lean_ctor_get(v___x_1434_, 0);
lean_inc(v_fst_1435_);
v_snd_1436_ = lean_ctor_get(v___x_1434_, 1);
lean_inc_n(v_snd_1436_, 2);
lean_dec_ref(v___x_1434_);
v___f_1437_ = ((lean_object*)(lp_aesop_Aesop_Frontend_modifyGetGlobalRuleSet___redArg___closed__0));
v___f_1438_ = ((lean_object*)(lp_aesop_Aesop_Frontend_modifyGetGlobalRuleSet___redArg___closed__1));
v___f_1439_ = ((lean_object*)(lp_aesop_Aesop_Frontend_modifyGetGlobalRuleSet___redArg___closed__2));
v___f_1440_ = lean_alloc_closure((void*)(lp_aesop_Aesop_Frontend_modifyGetGlobalRuleSet___redArg___lam__4___boxed), 2, 1);
lean_closure_set(v___f_1440_, 0, v_snd_1436_);
v___f_1441_ = lean_alloc_closure((void*)(lp_aesop_Aesop_Frontend_modifyGetGlobalRuleSet___redArg___lam__3___boxed), 2, 1);
lean_closure_set(v___f_1441_, 0, v_snd_1436_);
v_env_1442_ = l_Lean_ScopedEnvExtension_modifyState___redArg(v_fst_1413_, v_env_1423_, v___f_1439_);
v_env_1443_ = l_Lean_ScopedEnvExtension_modifyState___redArg(v_fst_1414_, v_env_1442_, v___f_1438_);
v_env_1444_ = l_Lean_ScopedEnvExtension_modifyState___redArg(v_snd_1415_, v_env_1443_, v___f_1437_);
v_env_1445_ = l_Lean_ScopedEnvExtension_modifyState___redArg(v_fst_1413_, v_env_1444_, v___f_1441_);
v_env_1446_ = l_Lean_ScopedEnvExtension_modifyState___redArg(v_fst_1414_, v_env_1445_, v___f_1440_);
v___x_1447_ = lp_aesop_Lean_setEnv___at___00Aesop_Frontend_modifyGetGlobalRuleSet___at___00Aesop_Frontend_modifyGlobalRuleSet_spec__0_spec__0___redArg(v_env_1446_, v___y_1406_);
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
lean_ctor_set(v___x_1449_, 0, v_fst_1435_);
v___x_1452_ = v___x_1449_;
goto v_reusejp_1451_;
}
else
{
lean_object* v_reuseFailAlloc_1453_; 
v_reuseFailAlloc_1453_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1453_, 0, v_fst_1435_);
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
lean_object* v_a_1456_; lean_object* v___x_1458_; uint8_t v_isShared_1459_; uint8_t v_isSharedCheck_1463_; 
lean_dec_ref(v_f_1404_);
v_a_1456_ = lean_ctor_get(v___x_1408_, 0);
v_isSharedCheck_1463_ = !lean_is_exclusive(v___x_1408_);
if (v_isSharedCheck_1463_ == 0)
{
v___x_1458_ = v___x_1408_;
v_isShared_1459_ = v_isSharedCheck_1463_;
goto v_resetjp_1457_;
}
else
{
lean_inc(v_a_1456_);
lean_dec(v___x_1408_);
v___x_1458_ = lean_box(0);
v_isShared_1459_ = v_isSharedCheck_1463_;
goto v_resetjp_1457_;
}
v_resetjp_1457_:
{
lean_object* v___x_1461_; 
if (v_isShared_1459_ == 0)
{
v___x_1461_ = v___x_1458_;
goto v_reusejp_1460_;
}
else
{
lean_object* v_reuseFailAlloc_1462_; 
v_reuseFailAlloc_1462_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1462_, 0, v_a_1456_);
v___x_1461_ = v_reuseFailAlloc_1462_;
goto v_reusejp_1460_;
}
v_reusejp_1460_:
{
return v___x_1461_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_modifyGetGlobalRuleSet___at___00Aesop_Frontend_modifyGlobalRuleSet_spec__0___redArg___boxed(lean_object* v_rsName_1464_, lean_object* v_f_1465_, lean_object* v___y_1466_, lean_object* v___y_1467_, lean_object* v___y_1468_){
_start:
{
lean_object* v_res_1469_; 
v_res_1469_ = lp_aesop_Aesop_Frontend_modifyGetGlobalRuleSet___at___00Aesop_Frontend_modifyGlobalRuleSet_spec__0___redArg(v_rsName_1464_, v_f_1465_, v___y_1466_, v___y_1467_);
lean_dec(v___y_1467_);
lean_dec_ref(v___y_1466_);
return v_res_1469_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_modifyGlobalRuleSet(lean_object* v_rsName_1470_, lean_object* v_f_1471_, lean_object* v_a_1472_, lean_object* v_a_1473_){
_start:
{
lean_object* v___f_1475_; lean_object* v___x_1476_; 
v___f_1475_ = lean_alloc_closure((void*)(lp_aesop_Aesop_Frontend_modifyGlobalRuleSet___lam__0), 2, 1);
lean_closure_set(v___f_1475_, 0, v_f_1471_);
v___x_1476_ = lp_aesop_Aesop_Frontend_modifyGetGlobalRuleSet___at___00Aesop_Frontend_modifyGlobalRuleSet_spec__0___redArg(v_rsName_1470_, v___f_1475_, v_a_1472_, v_a_1473_);
return v___x_1476_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_modifyGlobalRuleSet___boxed(lean_object* v_rsName_1477_, lean_object* v_f_1478_, lean_object* v_a_1479_, lean_object* v_a_1480_, lean_object* v_a_1481_){
_start:
{
lean_object* v_res_1482_; 
v_res_1482_ = lp_aesop_Aesop_Frontend_modifyGlobalRuleSet(v_rsName_1477_, v_f_1478_, v_a_1479_, v_a_1480_);
lean_dec(v_a_1480_);
lean_dec_ref(v_a_1479_);
return v_res_1482_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_setEnv___at___00Aesop_Frontend_modifyGetGlobalRuleSet___at___00Aesop_Frontend_modifyGlobalRuleSet_spec__0_spec__0(lean_object* v_env_1483_, lean_object* v___y_1484_, lean_object* v___y_1485_){
_start:
{
lean_object* v___x_1487_; 
v___x_1487_ = lp_aesop_Lean_setEnv___at___00Aesop_Frontend_modifyGetGlobalRuleSet___at___00Aesop_Frontend_modifyGlobalRuleSet_spec__0_spec__0___redArg(v_env_1483_, v___y_1485_);
return v___x_1487_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_setEnv___at___00Aesop_Frontend_modifyGetGlobalRuleSet___at___00Aesop_Frontend_modifyGlobalRuleSet_spec__0_spec__0___boxed(lean_object* v_env_1488_, lean_object* v___y_1489_, lean_object* v___y_1490_, lean_object* v___y_1491_){
_start:
{
lean_object* v_res_1492_; 
v_res_1492_ = lp_aesop_Lean_setEnv___at___00Aesop_Frontend_modifyGetGlobalRuleSet___at___00Aesop_Frontend_modifyGlobalRuleSet_spec__0_spec__0(v_env_1488_, v___y_1489_, v___y_1490_);
lean_dec(v___y_1490_);
lean_dec_ref(v___y_1489_);
return v_res_1492_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_modifyGetGlobalRuleSet___at___00Aesop_Frontend_modifyGlobalRuleSet_spec__0(lean_object* v_00_u03b1_1493_, lean_object* v_rsName_1494_, lean_object* v_f_1495_, lean_object* v___y_1496_, lean_object* v___y_1497_){
_start:
{
lean_object* v___x_1499_; 
v___x_1499_ = lp_aesop_Aesop_Frontend_modifyGetGlobalRuleSet___at___00Aesop_Frontend_modifyGlobalRuleSet_spec__0___redArg(v_rsName_1494_, v_f_1495_, v___y_1496_, v___y_1497_);
return v___x_1499_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_modifyGetGlobalRuleSet___at___00Aesop_Frontend_modifyGlobalRuleSet_spec__0___boxed(lean_object* v_00_u03b1_1500_, lean_object* v_rsName_1501_, lean_object* v_f_1502_, lean_object* v___y_1503_, lean_object* v___y_1504_, lean_object* v___y_1505_){
_start:
{
lean_object* v_res_1506_; 
v_res_1506_ = lp_aesop_Aesop_Frontend_modifyGetGlobalRuleSet___at___00Aesop_Frontend_modifyGlobalRuleSet_spec__0(v_00_u03b1_1500_, v_rsName_1501_, v_f_1502_, v___y_1503_, v___y_1504_);
lean_dec(v___y_1504_);
lean_dec_ref(v___y_1503_);
return v_res_1506_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_addGlobalRule___redArg___lam__0(lean_object* v_toPure_1507_, lean_object* v_____s_1508_){
_start:
{
lean_object* v___x_1509_; lean_object* v___x_1510_; 
v___x_1509_ = lean_box(0);
v___x_1510_ = lean_apply_2(v_toPure_1507_, lean_box(0), v___x_1509_);
return v___x_1510_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_addGlobalRule___redArg___lam__1(lean_object* v___x_1511_, lean_object* v_toPure_1512_, lean_object* v_r_1513_){
_start:
{
lean_object* v___x_1514_; lean_object* v___x_1515_; 
v___x_1514_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1514_, 0, v___x_1511_);
v___x_1515_ = lean_apply_2(v_toPure_1512_, lean_box(0), v___x_1514_);
return v___x_1515_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_addGlobalRule___redArg___lam__2(lean_object* v_a_1516_, lean_object* v___f_1517_, lean_object* v___f_1518_, lean_object* v_simpTheorems_1519_){
_start:
{
lean_object* v_pre_1520_; lean_object* v_post_1521_; lean_object* v_lemmaNames_1522_; lean_object* v_toUnfold_1523_; lean_object* v_erased_1524_; lean_object* v_toUnfoldThms_1525_; lean_object* v___x_1527_; uint8_t v_isShared_1528_; uint8_t v_isSharedCheck_1534_; 
v_pre_1520_ = lean_ctor_get(v_simpTheorems_1519_, 0);
v_post_1521_ = lean_ctor_get(v_simpTheorems_1519_, 1);
v_lemmaNames_1522_ = lean_ctor_get(v_simpTheorems_1519_, 2);
v_toUnfold_1523_ = lean_ctor_get(v_simpTheorems_1519_, 3);
v_erased_1524_ = lean_ctor_get(v_simpTheorems_1519_, 4);
v_toUnfoldThms_1525_ = lean_ctor_get(v_simpTheorems_1519_, 5);
v_isSharedCheck_1534_ = !lean_is_exclusive(v_simpTheorems_1519_);
if (v_isSharedCheck_1534_ == 0)
{
v___x_1527_ = v_simpTheorems_1519_;
v_isShared_1528_ = v_isSharedCheck_1534_;
goto v_resetjp_1526_;
}
else
{
lean_inc(v_toUnfoldThms_1525_);
lean_inc(v_erased_1524_);
lean_inc(v_toUnfold_1523_);
lean_inc(v_lemmaNames_1522_);
lean_inc(v_post_1521_);
lean_inc(v_pre_1520_);
lean_dec(v_simpTheorems_1519_);
v___x_1527_ = lean_box(0);
v_isShared_1528_ = v_isSharedCheck_1534_;
goto v_resetjp_1526_;
}
v_resetjp_1526_:
{
lean_object* v_origin_1529_; lean_object* v___x_1530_; lean_object* v___x_1532_; 
v_origin_1529_ = lean_ctor_get(v_a_1516_, 4);
lean_inc_ref(v_origin_1529_);
lean_dec_ref(v_a_1516_);
v___x_1530_ = l_Lean_PersistentHashMap_erase___redArg(v___f_1517_, v___f_1518_, v_erased_1524_, v_origin_1529_);
if (v_isShared_1528_ == 0)
{
lean_ctor_set(v___x_1527_, 4, v___x_1530_);
v___x_1532_ = v___x_1527_;
goto v_reusejp_1531_;
}
else
{
lean_object* v_reuseFailAlloc_1533_; 
v_reuseFailAlloc_1533_ = lean_alloc_ctor(0, 6, 0);
lean_ctor_set(v_reuseFailAlloc_1533_, 0, v_pre_1520_);
lean_ctor_set(v_reuseFailAlloc_1533_, 1, v_post_1521_);
lean_ctor_set(v_reuseFailAlloc_1533_, 2, v_lemmaNames_1522_);
lean_ctor_set(v_reuseFailAlloc_1533_, 3, v_toUnfold_1523_);
lean_ctor_set(v_reuseFailAlloc_1533_, 4, v___x_1530_);
lean_ctor_set(v_reuseFailAlloc_1533_, 5, v_toUnfoldThms_1525_);
v___x_1532_ = v_reuseFailAlloc_1533_;
goto v_reusejp_1531_;
}
v_reusejp_1531_:
{
return v___x_1532_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_addGlobalRule___redArg___lam__3(lean_object* v_fst_1535_, lean_object* v___f_1536_, lean_object* v_inst_1537_, lean_object* v_toBind_1538_, lean_object* v___f_1539_, lean_object* v_____do__lift_1540_){
_start:
{
lean_object* v___x_1541_; lean_object* v___x_1542_; lean_object* v___x_1543_; 
v___x_1541_ = l_Lean_ScopedEnvExtension_modifyState___redArg(v_fst_1535_, v_____do__lift_1540_, v___f_1536_);
v___x_1542_ = l_Lean_setEnv___redArg(v_inst_1537_, v___x_1541_);
v___x_1543_ = lean_apply_4(v_toBind_1538_, lean_box(0), lean_box(0), v___x_1542_, v___f_1539_);
return v___x_1543_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_addGlobalRule___redArg___lam__4(lean_object* v_a_1544_, lean_object* v_inst_1545_, lean_object* v___f_1546_, lean_object* v___f_1547_, lean_object* v_fst_1548_, lean_object* v_toBind_1549_, lean_object* v___f_1550_, lean_object* v___x_1551_, lean_object* v_toPure_1552_, lean_object* v_____r_1553_){
_start:
{
if (lean_obj_tag(v_a_1544_) == 0)
{
lean_object* v_a_1554_; lean_object* v_getEnv_1555_; lean_object* v___f_1556_; lean_object* v___f_1557_; lean_object* v___x_1558_; 
lean_dec(v_toPure_1552_);
v_a_1554_ = lean_ctor_get(v_a_1544_, 0);
lean_inc_ref(v_a_1554_);
lean_dec_ref_known(v_a_1544_, 1);
v_getEnv_1555_ = lean_ctor_get(v_inst_1545_, 0);
lean_inc(v_getEnv_1555_);
v___f_1556_ = lean_alloc_closure((void*)(lp_aesop_Aesop_Frontend_addGlobalRule___redArg___lam__2), 4, 3);
lean_closure_set(v___f_1556_, 0, v_a_1554_);
lean_closure_set(v___f_1556_, 1, v___f_1546_);
lean_closure_set(v___f_1556_, 2, v___f_1547_);
lean_inc(v_toBind_1549_);
v___f_1557_ = lean_alloc_closure((void*)(lp_aesop_Aesop_Frontend_addGlobalRule___redArg___lam__3), 6, 5);
lean_closure_set(v___f_1557_, 0, v_fst_1548_);
lean_closure_set(v___f_1557_, 1, v___f_1556_);
lean_closure_set(v___f_1557_, 2, v_inst_1545_);
lean_closure_set(v___f_1557_, 3, v_toBind_1549_);
lean_closure_set(v___f_1557_, 4, v___f_1550_);
v___x_1558_ = lean_apply_4(v_toBind_1549_, lean_box(0), lean_box(0), v_getEnv_1555_, v___f_1557_);
return v___x_1558_;
}
else
{
lean_object* v___x_1559_; lean_object* v___x_1560_; 
lean_dec(v___f_1550_);
lean_dec(v_toBind_1549_);
lean_dec_ref(v_fst_1548_);
lean_dec_ref(v___f_1547_);
lean_dec_ref(v___f_1546_);
lean_dec_ref(v_inst_1545_);
lean_dec_ref(v_a_1544_);
v___x_1559_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1559_, 0, v___x_1551_);
v___x_1560_ = lean_apply_2(v_toPure_1552_, lean_box(0), v___x_1559_);
return v___x_1560_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_addGlobalRule___redArg___lam__5(lean_object* v_inst_1561_, lean_object* v___f_1562_, lean_object* v___f_1563_, lean_object* v_fst_1564_, lean_object* v_toBind_1565_, lean_object* v___f_1566_, lean_object* v___x_1567_, lean_object* v_toPure_1568_, lean_object* v_inst_1569_, lean_object* v_inst_1570_, uint8_t v_kind_1571_, lean_object* v_a_1572_, lean_object* v_x_1573_, lean_object* v___y_1574_){
_start:
{
lean_object* v___f_1575_; lean_object* v___x_1576_; lean_object* v___x_1577_; 
lean_inc(v_toBind_1565_);
lean_inc_ref(v_fst_1564_);
lean_inc_ref(v_inst_1561_);
lean_inc_ref(v_a_1572_);
v___f_1575_ = lean_alloc_closure((void*)(lp_aesop_Aesop_Frontend_addGlobalRule___redArg___lam__4), 10, 9);
lean_closure_set(v___f_1575_, 0, v_a_1572_);
lean_closure_set(v___f_1575_, 1, v_inst_1561_);
lean_closure_set(v___f_1575_, 2, v___f_1562_);
lean_closure_set(v___f_1575_, 3, v___f_1563_);
lean_closure_set(v___f_1575_, 4, v_fst_1564_);
lean_closure_set(v___f_1575_, 5, v_toBind_1565_);
lean_closure_set(v___f_1575_, 6, v___f_1566_);
lean_closure_set(v___f_1575_, 7, v___x_1567_);
lean_closure_set(v___f_1575_, 8, v_toPure_1568_);
v___x_1576_ = l_Lean_ScopedEnvExtension_add___redArg(v_inst_1569_, v_inst_1570_, v_inst_1561_, v_fst_1564_, v_a_1572_, v_kind_1571_);
v___x_1577_ = lean_apply_4(v_toBind_1565_, lean_box(0), lean_box(0), v___x_1576_, v___f_1575_);
return v___x_1577_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_addGlobalRule___redArg___lam__5___boxed(lean_object* v_inst_1578_, lean_object* v___f_1579_, lean_object* v___f_1580_, lean_object* v_fst_1581_, lean_object* v_toBind_1582_, lean_object* v___f_1583_, lean_object* v___x_1584_, lean_object* v_toPure_1585_, lean_object* v_inst_1586_, lean_object* v_inst_1587_, lean_object* v_kind_1588_, lean_object* v_a_1589_, lean_object* v_x_1590_, lean_object* v___y_1591_){
_start:
{
uint8_t v_kind_boxed_1592_; lean_object* v_res_1593_; 
v_kind_boxed_1592_ = lean_unbox(v_kind_1588_);
v_res_1593_ = lp_aesop_Aesop_Frontend_addGlobalRule___redArg___lam__5(v_inst_1578_, v___f_1579_, v___f_1580_, v_fst_1581_, v_toBind_1582_, v___f_1583_, v___x_1584_, v_toPure_1585_, v_inst_1586_, v_inst_1587_, v_kind_boxed_1592_, v_a_1589_, v_x_1590_, v___y_1591_);
return v_res_1593_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_addGlobalRule___redArg___lam__6(lean_object* v_r_1594_, lean_object* v_inst_1595_, lean_object* v_inst_1596_, lean_object* v_inst_1597_, lean_object* v_fst_1598_, uint8_t v_kind_1599_, lean_object* v_toPure_1600_, lean_object* v___f_1601_, lean_object* v___f_1602_, lean_object* v_fst_1603_, lean_object* v_toBind_1604_, lean_object* v___f_1605_, lean_object* v_____r_1606_){
_start:
{
if (lean_obj_tag(v_r_1594_) == 0)
{
lean_object* v_m_1607_; lean_object* v___x_1608_; 
lean_dec(v___f_1605_);
lean_dec(v_toBind_1604_);
lean_dec_ref(v_fst_1603_);
lean_dec_ref(v___f_1602_);
lean_dec_ref(v___f_1601_);
lean_dec(v_toPure_1600_);
v_m_1607_ = lean_ctor_get(v_r_1594_, 0);
lean_inc_ref(v_m_1607_);
lean_dec_ref_known(v_r_1594_, 1);
v___x_1608_ = l_Lean_ScopedEnvExtension_add___redArg(v_inst_1595_, v_inst_1596_, v_inst_1597_, v_fst_1598_, v_m_1607_, v_kind_1599_);
return v___x_1608_;
}
else
{
lean_object* v_e_1609_; lean_object* v_entries_1610_; lean_object* v___x_1611_; lean_object* v___f_1612_; lean_object* v___x_1613_; lean_object* v___f_1614_; size_t v_sz_1615_; size_t v___x_1616_; lean_object* v___x_1617_; lean_object* v___x_1618_; 
lean_dec_ref(v_fst_1598_);
v_e_1609_ = lean_ctor_get(v_r_1594_, 0);
lean_inc_ref(v_e_1609_);
lean_dec_ref_known(v_r_1594_, 1);
v_entries_1610_ = lean_ctor_get(v_e_1609_, 1);
lean_inc_ref(v_entries_1610_);
lean_dec_ref(v_e_1609_);
v___x_1611_ = lean_box(0);
lean_inc(v_toPure_1600_);
v___f_1612_ = lean_alloc_closure((void*)(lp_aesop_Aesop_Frontend_addGlobalRule___redArg___lam__1), 3, 2);
lean_closure_set(v___f_1612_, 0, v___x_1611_);
lean_closure_set(v___f_1612_, 1, v_toPure_1600_);
v___x_1613_ = lean_box(v_kind_1599_);
lean_inc_ref(v_inst_1595_);
lean_inc(v_toBind_1604_);
v___f_1614_ = lean_alloc_closure((void*)(lp_aesop_Aesop_Frontend_addGlobalRule___redArg___lam__5___boxed), 14, 11);
lean_closure_set(v___f_1614_, 0, v_inst_1597_);
lean_closure_set(v___f_1614_, 1, v___f_1601_);
lean_closure_set(v___f_1614_, 2, v___f_1602_);
lean_closure_set(v___f_1614_, 3, v_fst_1603_);
lean_closure_set(v___f_1614_, 4, v_toBind_1604_);
lean_closure_set(v___f_1614_, 5, v___f_1612_);
lean_closure_set(v___f_1614_, 6, v___x_1611_);
lean_closure_set(v___f_1614_, 7, v_toPure_1600_);
lean_closure_set(v___f_1614_, 8, v_inst_1595_);
lean_closure_set(v___f_1614_, 9, v_inst_1596_);
lean_closure_set(v___f_1614_, 10, v___x_1613_);
v_sz_1615_ = lean_array_size(v_entries_1610_);
v___x_1616_ = ((size_t)0ULL);
v___x_1617_ = l___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop(lean_box(0), lean_box(0), lean_box(0), v_inst_1595_, v_entries_1610_, v___f_1614_, v_sz_1615_, v___x_1616_, v___x_1611_);
v___x_1618_ = lean_apply_4(v_toBind_1604_, lean_box(0), lean_box(0), v___x_1617_, v___f_1605_);
return v___x_1618_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_addGlobalRule___redArg___lam__6___boxed(lean_object* v_r_1619_, lean_object* v_inst_1620_, lean_object* v_inst_1621_, lean_object* v_inst_1622_, lean_object* v_fst_1623_, lean_object* v_kind_1624_, lean_object* v_toPure_1625_, lean_object* v___f_1626_, lean_object* v___f_1627_, lean_object* v_fst_1628_, lean_object* v_toBind_1629_, lean_object* v___f_1630_, lean_object* v_____r_1631_){
_start:
{
uint8_t v_kind_boxed_1632_; lean_object* v_res_1633_; 
v_kind_boxed_1632_ = lean_unbox(v_kind_1624_);
v_res_1633_ = lp_aesop_Aesop_Frontend_addGlobalRule___redArg___lam__6(v_r_1619_, v_inst_1620_, v_inst_1621_, v_inst_1622_, v_fst_1623_, v_kind_boxed_1632_, v_toPure_1625_, v___f_1626_, v___f_1627_, v_fst_1628_, v_toBind_1629_, v___f_1630_, v_____r_1631_);
return v_res_1633_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_addGlobalRule___redArg___lam__7(lean_object* v___f_1634_, lean_object* v_____r_1635_){
_start:
{
lean_object* v___x_1636_; 
v___x_1636_ = lean_apply_1(v___f_1634_, v_____r_1635_);
return v___x_1636_;
}
}
static lean_object* _init_lp_aesop_Aesop_Frontend_addGlobalRule___redArg___lam__8___closed__1(void){
_start:
{
lean_object* v___x_1638_; lean_object* v___x_1639_; 
v___x_1638_ = ((lean_object*)(lp_aesop_Aesop_Frontend_addGlobalRule___redArg___lam__8___closed__0));
v___x_1639_ = l_Lean_stringToMessageData(v___x_1638_);
return v___x_1639_;
}
}
static lean_object* _init_lp_aesop_Aesop_Frontend_addGlobalRule___redArg___lam__8___closed__3(void){
_start:
{
lean_object* v___x_1641_; lean_object* v___x_1642_; 
v___x_1641_ = ((lean_object*)(lp_aesop_Aesop_Frontend_addGlobalRule___redArg___lam__8___closed__2));
v___x_1642_ = l_Lean_stringToMessageData(v___x_1641_);
return v___x_1642_;
}
}
static lean_object* _init_lp_aesop_Aesop_Frontend_addGlobalRule___redArg___lam__8___closed__4(void){
_start:
{
lean_object* v___x_1643_; lean_object* v___x_1644_; 
v___x_1643_ = ((lean_object*)(lp_aesop_Aesop_Frontend_declareRuleSetUnchecked___closed__2));
v___x_1644_ = l_Lean_stringToMessageData(v___x_1643_);
return v___x_1644_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_addGlobalRule___redArg___lam__8(lean_object* v_r_1645_, lean_object* v___f_1646_, lean_object* v_rsName_1647_, lean_object* v_inst_1648_, lean_object* v_inst_1649_, lean_object* v_toBind_1650_, lean_object* v___f_1651_, lean_object* v_rs_1652_){
_start:
{
lean_object* v___x_1653_; uint8_t v___x_1654_; 
v___x_1653_ = lp_aesop_Aesop_GlobalRuleSetMember_name(v_r_1645_);
lean_inc_ref(v___x_1653_);
v___x_1654_ = lp_aesop_Aesop_GlobalRuleSet_contains(v_rs_1652_, v___x_1653_);
if (v___x_1654_ == 0)
{
lean_object* v___x_1655_; lean_object* v___x_1656_; 
lean_dec_ref(v___x_1653_);
lean_dec(v___f_1651_);
lean_dec(v_toBind_1650_);
lean_dec_ref(v_inst_1649_);
lean_dec_ref(v_inst_1648_);
lean_dec(v_rsName_1647_);
v___x_1655_ = lean_box(0);
v___x_1656_ = lean_apply_1(v___f_1646_, v___x_1655_);
return v___x_1656_;
}
else
{
lean_object* v_name_1657_; lean_object* v___x_1658_; lean_object* v___x_1659_; lean_object* v___x_1660_; lean_object* v___x_1661_; lean_object* v___x_1662_; lean_object* v___x_1663_; lean_object* v___x_1664_; lean_object* v___x_1665_; lean_object* v___x_1666_; lean_object* v___x_1667_; lean_object* v___x_1668_; 
lean_dec(v___f_1646_);
v_name_1657_ = lean_ctor_get(v___x_1653_, 0);
lean_inc(v_name_1657_);
lean_dec_ref(v___x_1653_);
v___x_1658_ = lean_obj_once(&lp_aesop_Aesop_Frontend_addGlobalRule___redArg___lam__8___closed__1, &lp_aesop_Aesop_Frontend_addGlobalRule___redArg___lam__8___closed__1_once, _init_lp_aesop_Aesop_Frontend_addGlobalRule___redArg___lam__8___closed__1);
v___x_1659_ = l_Lean_MessageData_ofName(v_name_1657_);
v___x_1660_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1660_, 0, v___x_1658_);
lean_ctor_set(v___x_1660_, 1, v___x_1659_);
v___x_1661_ = lean_obj_once(&lp_aesop_Aesop_Frontend_addGlobalRule___redArg___lam__8___closed__3, &lp_aesop_Aesop_Frontend_addGlobalRule___redArg___lam__8___closed__3_once, _init_lp_aesop_Aesop_Frontend_addGlobalRule___redArg___lam__8___closed__3);
v___x_1662_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1662_, 0, v___x_1660_);
lean_ctor_set(v___x_1662_, 1, v___x_1661_);
v___x_1663_ = l_Lean_MessageData_ofName(v_rsName_1647_);
v___x_1664_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1664_, 0, v___x_1662_);
lean_ctor_set(v___x_1664_, 1, v___x_1663_);
v___x_1665_ = lean_obj_once(&lp_aesop_Aesop_Frontend_addGlobalRule___redArg___lam__8___closed__4, &lp_aesop_Aesop_Frontend_addGlobalRule___redArg___lam__8___closed__4_once, _init_lp_aesop_Aesop_Frontend_addGlobalRule___redArg___lam__8___closed__4);
v___x_1666_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1666_, 0, v___x_1664_);
lean_ctor_set(v___x_1666_, 1, v___x_1665_);
v___x_1667_ = l_Lean_throwError___redArg(v_inst_1648_, v_inst_1649_, v___x_1666_);
v___x_1668_ = lean_apply_4(v_toBind_1650_, lean_box(0), lean_box(0), v___x_1667_, v___f_1651_);
return v___x_1668_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_addGlobalRule___redArg___lam__8___boxed(lean_object* v_r_1669_, lean_object* v___f_1670_, lean_object* v_rsName_1671_, lean_object* v_inst_1672_, lean_object* v_inst_1673_, lean_object* v_toBind_1674_, lean_object* v___f_1675_, lean_object* v_rs_1676_){
_start:
{
lean_object* v_res_1677_; 
v_res_1677_ = lp_aesop_Aesop_Frontend_addGlobalRule___redArg___lam__8(v_r_1669_, v___f_1670_, v_rsName_1671_, v_inst_1672_, v_inst_1673_, v_toBind_1674_, v___f_1675_, v_rs_1676_);
lean_dec_ref(v_rs_1676_);
lean_dec_ref(v_r_1669_);
return v_res_1677_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_addGlobalRule___redArg___lam__9(lean_object* v_r_1678_, lean_object* v_inst_1679_, lean_object* v_inst_1680_, lean_object* v_inst_1681_, uint8_t v_kind_1682_, lean_object* v_toPure_1683_, lean_object* v___f_1684_, lean_object* v___f_1685_, lean_object* v_toBind_1686_, lean_object* v___f_1687_, uint8_t v_checkNotExists_1688_, lean_object* v_rsName_1689_, lean_object* v_inst_1690_, lean_object* v_____x_1691_){
_start:
{
lean_object* v_snd_1692_; lean_object* v_snd_1693_; lean_object* v_snd_1694_; lean_object* v_fst_1695_; lean_object* v_fst_1696_; lean_object* v_snd_1697_; lean_object* v___x_1698_; lean_object* v___f_1699_; 
v_snd_1692_ = lean_ctor_get(v_____x_1691_, 1);
v_snd_1693_ = lean_ctor_get(v_snd_1692_, 1);
lean_inc(v_snd_1693_);
v_snd_1694_ = lean_ctor_get(v_snd_1693_, 1);
lean_inc(v_snd_1694_);
v_fst_1695_ = lean_ctor_get(v_____x_1691_, 0);
lean_inc_n(v_fst_1695_, 2);
lean_dec_ref(v_____x_1691_);
v_fst_1696_ = lean_ctor_get(v_snd_1693_, 0);
lean_inc_n(v_fst_1696_, 2);
lean_dec(v_snd_1693_);
v_snd_1697_ = lean_ctor_get(v_snd_1694_, 1);
lean_inc(v_snd_1697_);
lean_dec(v_snd_1694_);
v___x_1698_ = lean_box(v_kind_1682_);
lean_inc(v___f_1687_);
lean_inc(v_toBind_1686_);
lean_inc_ref(v___f_1685_);
lean_inc_ref(v___f_1684_);
lean_inc(v_toPure_1683_);
lean_inc_ref(v_inst_1681_);
lean_inc_ref(v_inst_1680_);
lean_inc_ref(v_inst_1679_);
lean_inc_ref(v_r_1678_);
v___f_1699_ = lean_alloc_closure((void*)(lp_aesop_Aesop_Frontend_addGlobalRule___redArg___lam__6___boxed), 13, 12);
lean_closure_set(v___f_1699_, 0, v_r_1678_);
lean_closure_set(v___f_1699_, 1, v_inst_1679_);
lean_closure_set(v___f_1699_, 2, v_inst_1680_);
lean_closure_set(v___f_1699_, 3, v_inst_1681_);
lean_closure_set(v___f_1699_, 4, v_fst_1695_);
lean_closure_set(v___f_1699_, 5, v___x_1698_);
lean_closure_set(v___f_1699_, 6, v_toPure_1683_);
lean_closure_set(v___f_1699_, 7, v___f_1684_);
lean_closure_set(v___f_1699_, 8, v___f_1685_);
lean_closure_set(v___f_1699_, 9, v_fst_1696_);
lean_closure_set(v___f_1699_, 10, v_toBind_1686_);
lean_closure_set(v___f_1699_, 11, v___f_1687_);
if (v_checkNotExists_1688_ == 0)
{
lean_object* v___x_1700_; lean_object* v___x_1701_; 
lean_dec_ref(v___f_1699_);
lean_dec(v_snd_1697_);
lean_dec_ref(v_inst_1690_);
lean_dec(v_rsName_1689_);
v___x_1700_ = lean_box(0);
v___x_1701_ = lp_aesop_Aesop_Frontend_addGlobalRule___redArg___lam__6(v_r_1678_, v_inst_1679_, v_inst_1680_, v_inst_1681_, v_fst_1695_, v_kind_1682_, v_toPure_1683_, v___f_1684_, v___f_1685_, v_fst_1696_, v_toBind_1686_, v___f_1687_, v___x_1700_);
return v___x_1701_;
}
else
{
lean_object* v___f_1702_; lean_object* v___f_1703_; lean_object* v___x_1704_; lean_object* v___x_1705_; 
lean_dec(v___f_1687_);
lean_dec_ref(v___f_1685_);
lean_dec_ref(v___f_1684_);
lean_dec(v_toPure_1683_);
lean_dec_ref(v_inst_1680_);
lean_inc_ref(v___f_1699_);
v___f_1702_ = lean_alloc_closure((void*)(lp_aesop_Aesop_Frontend_addGlobalRule___redArg___lam__7), 2, 1);
lean_closure_set(v___f_1702_, 0, v___f_1699_);
lean_inc(v_toBind_1686_);
lean_inc_ref(v_inst_1679_);
v___f_1703_ = lean_alloc_closure((void*)(lp_aesop_Aesop_Frontend_addGlobalRule___redArg___lam__8___boxed), 8, 7);
lean_closure_set(v___f_1703_, 0, v_r_1678_);
lean_closure_set(v___f_1703_, 1, v___f_1699_);
lean_closure_set(v___f_1703_, 2, v_rsName_1689_);
lean_closure_set(v___f_1703_, 3, v_inst_1679_);
lean_closure_set(v___f_1703_, 4, v_inst_1690_);
lean_closure_set(v___f_1703_, 5, v_toBind_1686_);
lean_closure_set(v___f_1703_, 6, v___f_1702_);
v___x_1704_ = lp_aesop_Aesop_Frontend_getGlobalRuleSetFromData___redArg(v_inst_1679_, v_inst_1681_, v_fst_1695_, v_fst_1696_, v_snd_1697_);
v___x_1705_ = lean_apply_4(v_toBind_1686_, lean_box(0), lean_box(0), v___x_1704_, v___f_1703_);
return v___x_1705_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_addGlobalRule___redArg___lam__9___boxed(lean_object* v_r_1706_, lean_object* v_inst_1707_, lean_object* v_inst_1708_, lean_object* v_inst_1709_, lean_object* v_kind_1710_, lean_object* v_toPure_1711_, lean_object* v___f_1712_, lean_object* v___f_1713_, lean_object* v_toBind_1714_, lean_object* v___f_1715_, lean_object* v_checkNotExists_1716_, lean_object* v_rsName_1717_, lean_object* v_inst_1718_, lean_object* v_____x_1719_){
_start:
{
uint8_t v_kind_boxed_1720_; uint8_t v_checkNotExists_boxed_1721_; lean_object* v_res_1722_; 
v_kind_boxed_1720_ = lean_unbox(v_kind_1710_);
v_checkNotExists_boxed_1721_ = lean_unbox(v_checkNotExists_1716_);
v_res_1722_ = lp_aesop_Aesop_Frontend_addGlobalRule___redArg___lam__9(v_r_1706_, v_inst_1707_, v_inst_1708_, v_inst_1709_, v_kind_boxed_1720_, v_toPure_1711_, v___f_1712_, v___f_1713_, v_toBind_1714_, v___f_1715_, v_checkNotExists_boxed_1721_, v_rsName_1717_, v_inst_1718_, v_____x_1719_);
return v_res_1722_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_addGlobalRule___redArg(lean_object* v_inst_1725_, lean_object* v_inst_1726_, lean_object* v_inst_1727_, lean_object* v_inst_1728_, lean_object* v_inst_1729_, lean_object* v_inst_1730_, lean_object* v_rsName_1731_, lean_object* v_r_1732_, uint8_t v_kind_1733_, uint8_t v_checkNotExists_1734_){
_start:
{
lean_object* v_toApplicative_1735_; lean_object* v_toBind_1736_; lean_object* v_toPure_1737_; lean_object* v___f_1738_; lean_object* v___f_1739_; lean_object* v___x_1740_; lean_object* v___f_1741_; lean_object* v___x_1742_; lean_object* v___x_1743_; lean_object* v___f_1744_; lean_object* v___x_1745_; 
v_toApplicative_1735_ = lean_ctor_get(v_inst_1725_, 0);
v_toBind_1736_ = lean_ctor_get(v_inst_1725_, 1);
lean_inc_n(v_toBind_1736_, 2);
v_toPure_1737_ = lean_ctor_get(v_toApplicative_1735_, 1);
lean_inc_n(v_toPure_1737_, 2);
v___f_1738_ = ((lean_object*)(lp_aesop_Aesop_Frontend_addGlobalRule___redArg___closed__0));
v___f_1739_ = ((lean_object*)(lp_aesop_Aesop_Frontend_addGlobalRule___redArg___closed__1));
lean_inc(v_rsName_1731_);
lean_inc_ref(v_inst_1726_);
lean_inc_ref(v_inst_1725_);
v___x_1740_ = lp_aesop_Aesop_Frontend_getGlobalRuleSetData___redArg(v_inst_1725_, v_inst_1726_, v_inst_1727_, v_inst_1728_, v_rsName_1731_);
v___f_1741_ = lean_alloc_closure((void*)(lp_aesop_Aesop_Frontend_addGlobalRule___redArg___lam__0), 2, 1);
lean_closure_set(v___f_1741_, 0, v_toPure_1737_);
v___x_1742_ = lean_box(v_kind_1733_);
v___x_1743_ = lean_box(v_checkNotExists_1734_);
v___f_1744_ = lean_alloc_closure((void*)(lp_aesop_Aesop_Frontend_addGlobalRule___redArg___lam__9___boxed), 14, 13);
lean_closure_set(v___f_1744_, 0, v_r_1732_);
lean_closure_set(v___f_1744_, 1, v_inst_1725_);
lean_closure_set(v___f_1744_, 2, v_inst_1730_);
lean_closure_set(v___f_1744_, 3, v_inst_1729_);
lean_closure_set(v___f_1744_, 4, v___x_1742_);
lean_closure_set(v___f_1744_, 5, v_toPure_1737_);
lean_closure_set(v___f_1744_, 6, v___f_1738_);
lean_closure_set(v___f_1744_, 7, v___f_1739_);
lean_closure_set(v___f_1744_, 8, v_toBind_1736_);
lean_closure_set(v___f_1744_, 9, v___f_1741_);
lean_closure_set(v___f_1744_, 10, v___x_1743_);
lean_closure_set(v___f_1744_, 11, v_rsName_1731_);
lean_closure_set(v___f_1744_, 12, v_inst_1726_);
v___x_1745_ = lean_apply_4(v_toBind_1736_, lean_box(0), lean_box(0), v___x_1740_, v___f_1744_);
return v___x_1745_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_addGlobalRule___redArg___boxed(lean_object* v_inst_1746_, lean_object* v_inst_1747_, lean_object* v_inst_1748_, lean_object* v_inst_1749_, lean_object* v_inst_1750_, lean_object* v_inst_1751_, lean_object* v_rsName_1752_, lean_object* v_r_1753_, lean_object* v_kind_1754_, lean_object* v_checkNotExists_1755_){
_start:
{
uint8_t v_kind_boxed_1756_; uint8_t v_checkNotExists_boxed_1757_; lean_object* v_res_1758_; 
v_kind_boxed_1756_ = lean_unbox(v_kind_1754_);
v_checkNotExists_boxed_1757_ = lean_unbox(v_checkNotExists_1755_);
v_res_1758_ = lp_aesop_Aesop_Frontend_addGlobalRule___redArg(v_inst_1746_, v_inst_1747_, v_inst_1748_, v_inst_1749_, v_inst_1750_, v_inst_1751_, v_rsName_1752_, v_r_1753_, v_kind_boxed_1756_, v_checkNotExists_boxed_1757_);
return v_res_1758_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_addGlobalRule(lean_object* v_m_1759_, lean_object* v_inst_1760_, lean_object* v_inst_1761_, lean_object* v_inst_1762_, lean_object* v_inst_1763_, lean_object* v_inst_1764_, lean_object* v_inst_1765_, lean_object* v_rsName_1766_, lean_object* v_r_1767_, uint8_t v_kind_1768_, uint8_t v_checkNotExists_1769_){
_start:
{
lean_object* v___x_1770_; 
v___x_1770_ = lp_aesop_Aesop_Frontend_addGlobalRule___redArg(v_inst_1760_, v_inst_1761_, v_inst_1762_, v_inst_1763_, v_inst_1764_, v_inst_1765_, v_rsName_1766_, v_r_1767_, v_kind_1768_, v_checkNotExists_1769_);
return v___x_1770_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_addGlobalRule___boxed(lean_object* v_m_1771_, lean_object* v_inst_1772_, lean_object* v_inst_1773_, lean_object* v_inst_1774_, lean_object* v_inst_1775_, lean_object* v_inst_1776_, lean_object* v_inst_1777_, lean_object* v_rsName_1778_, lean_object* v_r_1779_, lean_object* v_kind_1780_, lean_object* v_checkNotExists_1781_){
_start:
{
uint8_t v_kind_boxed_1782_; uint8_t v_checkNotExists_boxed_1783_; lean_object* v_res_1784_; 
v_kind_boxed_1782_ = lean_unbox(v_kind_1780_);
v_checkNotExists_boxed_1783_ = lean_unbox(v_checkNotExists_1781_);
v_res_1784_ = lp_aesop_Aesop_Frontend_addGlobalRule(v_m_1771_, v_inst_1772_, v_inst_1773_, v_inst_1774_, v_inst_1775_, v_inst_1776_, v_inst_1777_, v_rsName_1778_, v_r_1779_, v_kind_boxed_1782_, v_checkNotExists_boxed_1783_);
return v_res_1784_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Frontend_Extension_0__Aesop_Frontend_eraseGlobalRules_go___redArg___lam__0(lean_object* v_rf_1785_, uint8_t v_anyErased_1786_, lean_object* v_rs_1787_){
_start:
{
lean_object* v___x_1788_; 
v___x_1788_ = lp_aesop_Aesop_GlobalRuleSet_erase(v_rs_1787_, v_rf_1785_);
if (v_anyErased_1786_ == 0)
{
lean_object* v_fst_1789_; lean_object* v_snd_1790_; lean_object* v___x_1792_; uint8_t v_isShared_1793_; uint8_t v_isSharedCheck_1797_; 
v_fst_1789_ = lean_ctor_get(v___x_1788_, 0);
v_snd_1790_ = lean_ctor_get(v___x_1788_, 1);
v_isSharedCheck_1797_ = !lean_is_exclusive(v___x_1788_);
if (v_isSharedCheck_1797_ == 0)
{
v___x_1792_ = v___x_1788_;
v_isShared_1793_ = v_isSharedCheck_1797_;
goto v_resetjp_1791_;
}
else
{
lean_inc(v_snd_1790_);
lean_inc(v_fst_1789_);
lean_dec(v___x_1788_);
v___x_1792_ = lean_box(0);
v_isShared_1793_ = v_isSharedCheck_1797_;
goto v_resetjp_1791_;
}
v_resetjp_1791_:
{
lean_object* v___x_1795_; 
if (v_isShared_1793_ == 0)
{
lean_ctor_set(v___x_1792_, 1, v_fst_1789_);
lean_ctor_set(v___x_1792_, 0, v_snd_1790_);
v___x_1795_ = v___x_1792_;
goto v_reusejp_1794_;
}
else
{
lean_object* v_reuseFailAlloc_1796_; 
v_reuseFailAlloc_1796_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1796_, 0, v_snd_1790_);
lean_ctor_set(v_reuseFailAlloc_1796_, 1, v_fst_1789_);
v___x_1795_ = v_reuseFailAlloc_1796_;
goto v_reusejp_1794_;
}
v_reusejp_1794_:
{
return v___x_1795_;
}
}
}
else
{
lean_object* v_fst_1798_; lean_object* v___x_1800_; uint8_t v_isShared_1801_; uint8_t v_isSharedCheck_1806_; 
v_fst_1798_ = lean_ctor_get(v___x_1788_, 0);
v_isSharedCheck_1806_ = !lean_is_exclusive(v___x_1788_);
if (v_isSharedCheck_1806_ == 0)
{
lean_object* v_unused_1807_; 
v_unused_1807_ = lean_ctor_get(v___x_1788_, 1);
lean_dec(v_unused_1807_);
v___x_1800_ = v___x_1788_;
v_isShared_1801_ = v_isSharedCheck_1806_;
goto v_resetjp_1799_;
}
else
{
lean_inc(v_fst_1798_);
lean_dec(v___x_1788_);
v___x_1800_ = lean_box(0);
v_isShared_1801_ = v_isSharedCheck_1806_;
goto v_resetjp_1799_;
}
v_resetjp_1799_:
{
lean_object* v___x_1802_; lean_object* v___x_1804_; 
v___x_1802_ = lean_box(v_anyErased_1786_);
if (v_isShared_1801_ == 0)
{
lean_ctor_set(v___x_1800_, 1, v_fst_1798_);
lean_ctor_set(v___x_1800_, 0, v___x_1802_);
v___x_1804_ = v___x_1800_;
goto v_reusejp_1803_;
}
else
{
lean_object* v_reuseFailAlloc_1805_; 
v_reuseFailAlloc_1805_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1805_, 0, v___x_1802_);
lean_ctor_set(v_reuseFailAlloc_1805_, 1, v_fst_1798_);
v___x_1804_ = v_reuseFailAlloc_1805_;
goto v_reusejp_1803_;
}
v_reusejp_1803_:
{
return v___x_1804_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Frontend_Extension_0__Aesop_Frontend_eraseGlobalRules_go___redArg___lam__0___boxed(lean_object* v_rf_1808_, lean_object* v_anyErased_1809_, lean_object* v_rs_1810_){
_start:
{
uint8_t v_anyErased_boxed_1811_; lean_object* v_res_1812_; 
v_anyErased_boxed_1811_ = lean_unbox(v_anyErased_1809_);
v_res_1812_ = lp_aesop___private_Aesop_Frontend_Extension_0__Aesop_Frontend_eraseGlobalRules_go___redArg___lam__0(v_rf_1808_, v_anyErased_boxed_1811_, v_rs_1810_);
return v_res_1812_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Frontend_Extension_0__Aesop_Frontend_eraseGlobalRules_go___redArg(lean_object* v_inst_1813_, lean_object* v_inst_1814_, lean_object* v_inst_1815_, lean_object* v_inst_1816_, lean_object* v_inst_1817_, lean_object* v_rf_1818_, uint8_t v_anyErased_1819_, lean_object* v_rsName_1820_){
_start:
{
lean_object* v___x_1821_; lean_object* v___f_1822_; lean_object* v___x_1823_; 
v___x_1821_ = lean_box(v_anyErased_1819_);
v___f_1822_ = lean_alloc_closure((void*)(lp_aesop___private_Aesop_Frontend_Extension_0__Aesop_Frontend_eraseGlobalRules_go___redArg___lam__0___boxed), 3, 2);
lean_closure_set(v___f_1822_, 0, v_rf_1818_);
lean_closure_set(v___f_1822_, 1, v___x_1821_);
v___x_1823_ = lp_aesop_Aesop_Frontend_modifyGetGlobalRuleSet___redArg(v_inst_1813_, v_inst_1814_, v_inst_1815_, v_inst_1816_, v_inst_1817_, v_rsName_1820_, v___f_1822_);
return v___x_1823_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Frontend_Extension_0__Aesop_Frontend_eraseGlobalRules_go___redArg___boxed(lean_object* v_inst_1824_, lean_object* v_inst_1825_, lean_object* v_inst_1826_, lean_object* v_inst_1827_, lean_object* v_inst_1828_, lean_object* v_rf_1829_, lean_object* v_anyErased_1830_, lean_object* v_rsName_1831_){
_start:
{
uint8_t v_anyErased_boxed_1832_; lean_object* v_res_1833_; 
v_anyErased_boxed_1832_ = lean_unbox(v_anyErased_1830_);
v_res_1833_ = lp_aesop___private_Aesop_Frontend_Extension_0__Aesop_Frontend_eraseGlobalRules_go___redArg(v_inst_1824_, v_inst_1825_, v_inst_1826_, v_inst_1827_, v_inst_1828_, v_rf_1829_, v_anyErased_boxed_1832_, v_rsName_1831_);
return v_res_1833_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Frontend_Extension_0__Aesop_Frontend_eraseGlobalRules_go(lean_object* v_m_1834_, lean_object* v_inst_1835_, lean_object* v_inst_1836_, lean_object* v_inst_1837_, lean_object* v_inst_1838_, lean_object* v_inst_1839_, lean_object* v_rf_1840_, uint8_t v_anyErased_1841_, lean_object* v_rsName_1842_){
_start:
{
lean_object* v___x_1843_; 
v___x_1843_ = lp_aesop___private_Aesop_Frontend_Extension_0__Aesop_Frontend_eraseGlobalRules_go___redArg(v_inst_1835_, v_inst_1836_, v_inst_1837_, v_inst_1838_, v_inst_1839_, v_rf_1840_, v_anyErased_1841_, v_rsName_1842_);
return v___x_1843_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Frontend_Extension_0__Aesop_Frontend_eraseGlobalRules_go___boxed(lean_object* v_m_1844_, lean_object* v_inst_1845_, lean_object* v_inst_1846_, lean_object* v_inst_1847_, lean_object* v_inst_1848_, lean_object* v_inst_1849_, lean_object* v_rf_1850_, lean_object* v_anyErased_1851_, lean_object* v_rsName_1852_){
_start:
{
uint8_t v_anyErased_boxed_1853_; lean_object* v_res_1854_; 
v_anyErased_boxed_1853_ = lean_unbox(v_anyErased_1851_);
v_res_1854_ = lp_aesop___private_Aesop_Frontend_Extension_0__Aesop_Frontend_eraseGlobalRules_go(v_m_1844_, v_inst_1845_, v_inst_1846_, v_inst_1847_, v_inst_1848_, v_inst_1849_, v_rf_1850_, v_anyErased_boxed_1853_, v_rsName_1852_);
return v_res_1854_;
}
}
static lean_object* _init_lp_aesop_Aesop_Frontend_eraseGlobalRules___redArg___lam__0___closed__1(void){
_start:
{
lean_object* v___x_1856_; lean_object* v___x_1857_; 
v___x_1856_ = ((lean_object*)(lp_aesop_Aesop_Frontend_eraseGlobalRules___redArg___lam__0___closed__0));
v___x_1857_ = l_Lean_stringToMessageData(v___x_1856_);
return v___x_1857_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_eraseGlobalRules___redArg___lam__0(lean_object* v_toApplicative_1858_, uint8_t v_checkExists_1859_, lean_object* v_rf_1860_, lean_object* v_inst_1861_, lean_object* v_inst_1862_, uint8_t v_anyErased_1863_){
_start:
{
if (v_checkExists_1859_ == 0)
{
lean_dec_ref(v_inst_1862_);
lean_dec_ref(v_inst_1861_);
lean_dec_ref(v_rf_1860_);
goto v___jp_1864_;
}
else
{
if (v_anyErased_1863_ == 0)
{
lean_object* v_name_1868_; lean_object* v___x_1869_; lean_object* v___x_1870_; lean_object* v___x_1871_; lean_object* v___x_1872_; lean_object* v___x_1873_; lean_object* v___x_1874_; 
lean_dec_ref(v_toApplicative_1858_);
v_name_1868_ = lean_ctor_get(v_rf_1860_, 0);
lean_inc(v_name_1868_);
lean_dec_ref(v_rf_1860_);
v___x_1869_ = lean_obj_once(&lp_aesop_Aesop_Frontend_addGlobalRule___redArg___lam__8___closed__4, &lp_aesop_Aesop_Frontend_addGlobalRule___redArg___lam__8___closed__4_once, _init_lp_aesop_Aesop_Frontend_addGlobalRule___redArg___lam__8___closed__4);
v___x_1870_ = l_Lean_MessageData_ofName(v_name_1868_);
v___x_1871_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1871_, 0, v___x_1869_);
lean_ctor_set(v___x_1871_, 1, v___x_1870_);
v___x_1872_ = lean_obj_once(&lp_aesop_Aesop_Frontend_eraseGlobalRules___redArg___lam__0___closed__1, &lp_aesop_Aesop_Frontend_eraseGlobalRules___redArg___lam__0___closed__1_once, _init_lp_aesop_Aesop_Frontend_eraseGlobalRules___redArg___lam__0___closed__1);
v___x_1873_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1873_, 0, v___x_1871_);
lean_ctor_set(v___x_1873_, 1, v___x_1872_);
v___x_1874_ = l_Lean_throwError___redArg(v_inst_1861_, v_inst_1862_, v___x_1873_);
return v___x_1874_;
}
else
{
lean_dec_ref(v_inst_1862_);
lean_dec_ref(v_inst_1861_);
lean_dec_ref(v_rf_1860_);
goto v___jp_1864_;
}
}
v___jp_1864_:
{
lean_object* v_toPure_1865_; lean_object* v___x_1866_; lean_object* v___x_1867_; 
v_toPure_1865_ = lean_ctor_get(v_toApplicative_1858_, 1);
lean_inc(v_toPure_1865_);
lean_dec_ref(v_toApplicative_1858_);
v___x_1866_ = lean_box(0);
v___x_1867_ = lean_apply_2(v_toPure_1865_, lean_box(0), v___x_1866_);
return v___x_1867_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_eraseGlobalRules___redArg___lam__0___boxed(lean_object* v_toApplicative_1875_, lean_object* v_checkExists_1876_, lean_object* v_rf_1877_, lean_object* v_inst_1878_, lean_object* v_inst_1879_, lean_object* v_anyErased_1880_){
_start:
{
uint8_t v_checkExists_boxed_1881_; uint8_t v_anyErased_boxed_1882_; lean_object* v_res_1883_; 
v_checkExists_boxed_1881_ = lean_unbox(v_checkExists_1876_);
v_anyErased_boxed_1882_ = lean_unbox(v_anyErased_1880_);
v_res_1883_ = lp_aesop_Aesop_Frontend_eraseGlobalRules___redArg___lam__0(v_toApplicative_1875_, v_checkExists_boxed_1881_, v_rf_1877_, v_inst_1878_, v_inst_1879_, v_anyErased_boxed_1882_);
return v_res_1883_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_eraseGlobalRules___redArg___lam__1(lean_object* v_inst_1884_, lean_object* v_inst_1885_, lean_object* v_inst_1886_, lean_object* v_inst_1887_, lean_object* v_inst_1888_, lean_object* v_rf_1889_, uint8_t v_b_1890_, lean_object* v_rsName_1891_, lean_object* v_x_1892_){
_start:
{
lean_object* v___x_1893_; 
v___x_1893_ = lp_aesop___private_Aesop_Frontend_Extension_0__Aesop_Frontend_eraseGlobalRules_go___redArg(v_inst_1884_, v_inst_1885_, v_inst_1886_, v_inst_1887_, v_inst_1888_, v_rf_1889_, v_b_1890_, v_rsName_1891_);
return v___x_1893_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_eraseGlobalRules___redArg___lam__1___boxed(lean_object* v_inst_1894_, lean_object* v_inst_1895_, lean_object* v_inst_1896_, lean_object* v_inst_1897_, lean_object* v_inst_1898_, lean_object* v_rf_1899_, lean_object* v_b_1900_, lean_object* v_rsName_1901_, lean_object* v_x_1902_){
_start:
{
uint8_t v_b_boxed_1903_; lean_object* v_res_1904_; 
v_b_boxed_1903_ = lean_unbox(v_b_1900_);
v_res_1904_ = lp_aesop_Aesop_Frontend_eraseGlobalRules___redArg___lam__1(v_inst_1894_, v_inst_1895_, v_inst_1896_, v_inst_1897_, v_inst_1898_, v_rf_1899_, v_b_boxed_1903_, v_rsName_1901_, v_x_1902_);
lean_dec_ref(v_x_1902_);
return v_res_1904_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_eraseGlobalRules___redArg___lam__2(lean_object* v_inst_1905_, lean_object* v___f_1906_, uint8_t v_acc_1907_, lean_object* v_l_1908_){
_start:
{
lean_object* v___x_1909_; lean_object* v___x_1910_; 
v___x_1909_ = lean_box(v_acc_1907_);
v___x_1910_ = l_Std_DHashMap_Internal_AssocList_foldlM___redArg(v_inst_1905_, v___f_1906_, v___x_1909_, v_l_1908_);
return v___x_1910_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_eraseGlobalRules___redArg___lam__2___boxed(lean_object* v_inst_1911_, lean_object* v___f_1912_, lean_object* v_acc_1913_, lean_object* v_l_1914_){
_start:
{
uint8_t v_acc_boxed_1915_; lean_object* v_res_1916_; 
v_acc_boxed_1915_ = lean_unbox(v_acc_1913_);
v_res_1916_ = lp_aesop_Aesop_Frontend_eraseGlobalRules___redArg___lam__2(v_inst_1911_, v___f_1912_, v_acc_boxed_1915_, v_l_1914_);
return v_res_1916_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_eraseGlobalRules___redArg___lam__3(lean_object* v_toApplicative_1917_, lean_object* v_toBind_1918_, lean_object* v___f_1919_, lean_object* v_inst_1920_, lean_object* v___f_1921_, lean_object* v_____do__lift_1922_){
_start:
{
lean_object* v_buckets_1923_; uint8_t v___x_1924_; lean_object* v___x_1925_; lean_object* v___x_1926_; uint8_t v___x_1927_; 
v_buckets_1923_ = lean_ctor_get(v_____do__lift_1922_, 1);
lean_inc_ref(v_buckets_1923_);
lean_dec_ref(v_____do__lift_1922_);
v___x_1924_ = 0;
v___x_1925_ = lean_unsigned_to_nat(0u);
v___x_1926_ = lean_array_get_size(v_buckets_1923_);
v___x_1927_ = lean_nat_dec_lt(v___x_1925_, v___x_1926_);
if (v___x_1927_ == 0)
{
lean_object* v_toPure_1928_; lean_object* v___x_1929_; lean_object* v___x_1930_; lean_object* v___x_1931_; 
lean_dec_ref(v_buckets_1923_);
lean_dec(v___f_1921_);
lean_dec_ref(v_inst_1920_);
v_toPure_1928_ = lean_ctor_get(v_toApplicative_1917_, 1);
lean_inc(v_toPure_1928_);
lean_dec_ref(v_toApplicative_1917_);
v___x_1929_ = lean_box(v___x_1924_);
v___x_1930_ = lean_apply_2(v_toPure_1928_, lean_box(0), v___x_1929_);
v___x_1931_ = lean_apply_4(v_toBind_1918_, lean_box(0), lean_box(0), v___x_1930_, v___f_1919_);
return v___x_1931_;
}
else
{
uint8_t v___x_1932_; 
v___x_1932_ = lean_nat_dec_le(v___x_1926_, v___x_1926_);
if (v___x_1932_ == 0)
{
if (v___x_1927_ == 0)
{
lean_object* v_toPure_1933_; lean_object* v___x_1934_; lean_object* v___x_1935_; lean_object* v___x_1936_; 
lean_dec_ref(v_buckets_1923_);
lean_dec(v___f_1921_);
lean_dec_ref(v_inst_1920_);
v_toPure_1933_ = lean_ctor_get(v_toApplicative_1917_, 1);
lean_inc(v_toPure_1933_);
lean_dec_ref(v_toApplicative_1917_);
v___x_1934_ = lean_box(v___x_1924_);
v___x_1935_ = lean_apply_2(v_toPure_1933_, lean_box(0), v___x_1934_);
v___x_1936_ = lean_apply_4(v_toBind_1918_, lean_box(0), lean_box(0), v___x_1935_, v___f_1919_);
return v___x_1936_;
}
else
{
size_t v___x_1937_; size_t v___x_1938_; lean_object* v___x_1939_; lean_object* v___x_1940_; lean_object* v___x_1941_; 
lean_dec_ref(v_toApplicative_1917_);
v___x_1937_ = ((size_t)0ULL);
v___x_1938_ = lean_usize_of_nat(v___x_1926_);
v___x_1939_ = lean_box(v___x_1924_);
v___x_1940_ = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), v_inst_1920_, v___f_1921_, v_buckets_1923_, v___x_1937_, v___x_1938_, v___x_1939_);
v___x_1941_ = lean_apply_4(v_toBind_1918_, lean_box(0), lean_box(0), v___x_1940_, v___f_1919_);
return v___x_1941_;
}
}
else
{
size_t v___x_1942_; size_t v___x_1943_; lean_object* v___x_1944_; lean_object* v___x_1945_; lean_object* v___x_1946_; 
lean_dec_ref(v_toApplicative_1917_);
v___x_1942_ = ((size_t)0ULL);
v___x_1943_ = lean_usize_of_nat(v___x_1926_);
v___x_1944_ = lean_box(v___x_1924_);
v___x_1945_ = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), v_inst_1920_, v___f_1921_, v_buckets_1923_, v___x_1942_, v___x_1943_, v___x_1944_);
v___x_1946_ = lean_apply_4(v_toBind_1918_, lean_box(0), lean_box(0), v___x_1945_, v___f_1919_);
return v___x_1946_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_eraseGlobalRules___redArg___lam__4(uint8_t v_checkExists_1947_, lean_object* v_x_1948_){
_start:
{
lean_object* v___x_1949_; 
v___x_1949_ = l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(v_x_1948_, v_checkExists_1947_);
return v___x_1949_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_eraseGlobalRules___redArg___lam__4___boxed(lean_object* v_checkExists_1950_, lean_object* v_x_1951_){
_start:
{
uint8_t v_checkExists_boxed_1952_; lean_object* v_res_1953_; 
v_checkExists_boxed_1952_ = lean_unbox(v_checkExists_1950_);
v_res_1953_ = lp_aesop_Aesop_Frontend_eraseGlobalRules___redArg___lam__4(v_checkExists_boxed_1952_, v_x_1951_);
return v_res_1953_;
}
}
static lean_object* _init_lp_aesop_Aesop_Frontend_eraseGlobalRules___redArg___lam__5___closed__1(void){
_start:
{
lean_object* v___x_1955_; lean_object* v___x_1956_; 
v___x_1955_ = ((lean_object*)(lp_aesop_Aesop_Frontend_eraseGlobalRules___redArg___lam__5___closed__0));
v___x_1956_ = l_Lean_stringToMessageData(v___x_1955_);
return v___x_1956_;
}
}
static lean_object* _init_lp_aesop_Aesop_Frontend_eraseGlobalRules___redArg___lam__5___closed__14(void){
_start:
{
lean_object* v___x_1978_; lean_object* v___x_1979_; 
v___x_1978_ = ((lean_object*)(lp_aesop_Aesop_Frontend_eraseGlobalRules___redArg___lam__5___closed__13));
v___x_1979_ = l_Lean_stringToMessageData(v___x_1978_);
return v___x_1979_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_eraseGlobalRules___redArg___lam__5(lean_object* v_toApplicative_1980_, uint8_t v_checkExists_1981_, lean_object* v_rf_1982_, lean_object* v_val_1983_, lean_object* v___f_1984_, lean_object* v_inst_1985_, lean_object* v_inst_1986_, uint8_t v_anyErased_1987_){
_start:
{
if (v_checkExists_1981_ == 0)
{
lean_dec_ref(v_inst_1986_);
lean_dec_ref(v_inst_1985_);
lean_dec_ref(v___f_1984_);
lean_dec_ref(v_val_1983_);
lean_dec_ref(v_rf_1982_);
goto v___jp_1988_;
}
else
{
if (v_anyErased_1987_ == 0)
{
lean_object* v_name_1992_; lean_object* v___x_1993_; lean_object* v___x_1994_; lean_object* v___x_1995_; lean_object* v___x_1996_; lean_object* v___x_1997_; lean_object* v___x_1998_; size_t v_sz_1999_; size_t v___x_2000_; lean_object* v___x_2001_; lean_object* v___x_2002_; lean_object* v___x_2003_; lean_object* v___x_2004_; lean_object* v___x_2005_; lean_object* v___x_2006_; lean_object* v___x_2007_; lean_object* v___x_2008_; lean_object* v___x_2009_; lean_object* v___x_2010_; 
lean_dec_ref(v_toApplicative_1980_);
v_name_1992_ = lean_ctor_get(v_rf_1982_, 0);
lean_inc(v_name_1992_);
lean_dec_ref(v_rf_1982_);
v___x_1993_ = lean_obj_once(&lp_aesop_Aesop_Frontend_addGlobalRule___redArg___lam__8___closed__4, &lp_aesop_Aesop_Frontend_addGlobalRule___redArg___lam__8___closed__4_once, _init_lp_aesop_Aesop_Frontend_addGlobalRule___redArg___lam__8___closed__4);
v___x_1994_ = l_Lean_MessageData_ofName(v_name_1992_);
v___x_1995_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1995_, 0, v___x_1993_);
lean_ctor_set(v___x_1995_, 1, v___x_1994_);
v___x_1996_ = lean_obj_once(&lp_aesop_Aesop_Frontend_eraseGlobalRules___redArg___lam__5___closed__1, &lp_aesop_Aesop_Frontend_eraseGlobalRules___redArg___lam__5___closed__1_once, _init_lp_aesop_Aesop_Frontend_eraseGlobalRules___redArg___lam__5___closed__1);
v___x_1997_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1997_, 0, v___x_1995_);
lean_ctor_set(v___x_1997_, 1, v___x_1996_);
v___x_1998_ = ((lean_object*)(lp_aesop_Aesop_Frontend_eraseGlobalRules___redArg___lam__5___closed__11));
v_sz_1999_ = lean_array_size(v_val_1983_);
v___x_2000_ = ((size_t)0ULL);
v___x_2001_ = l___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map(lean_box(0), lean_box(0), lean_box(0), v___x_1998_, v___f_1984_, v_sz_1999_, v___x_2000_, v_val_1983_);
v___x_2002_ = lean_array_to_list(v___x_2001_);
v___x_2003_ = ((lean_object*)(lp_aesop_Aesop_Frontend_eraseGlobalRules___redArg___lam__5___closed__12));
v___x_2004_ = lean_box(0);
v___x_2005_ = l_List_mapTR_loop___redArg(v___x_2003_, v___x_2002_, v___x_2004_);
v___x_2006_ = l_Lean_MessageData_ofList(v___x_2005_);
v___x_2007_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2007_, 0, v___x_1997_);
lean_ctor_set(v___x_2007_, 1, v___x_2006_);
v___x_2008_ = lean_obj_once(&lp_aesop_Aesop_Frontend_eraseGlobalRules___redArg___lam__5___closed__14, &lp_aesop_Aesop_Frontend_eraseGlobalRules___redArg___lam__5___closed__14_once, _init_lp_aesop_Aesop_Frontend_eraseGlobalRules___redArg___lam__5___closed__14);
v___x_2009_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2009_, 0, v___x_2007_);
lean_ctor_set(v___x_2009_, 1, v___x_2008_);
v___x_2010_ = l_Lean_throwError___redArg(v_inst_1985_, v_inst_1986_, v___x_2009_);
return v___x_2010_;
}
else
{
lean_dec_ref(v_inst_1986_);
lean_dec_ref(v_inst_1985_);
lean_dec_ref(v___f_1984_);
lean_dec_ref(v_val_1983_);
lean_dec_ref(v_rf_1982_);
goto v___jp_1988_;
}
}
v___jp_1988_:
{
lean_object* v_toPure_1989_; lean_object* v___x_1990_; lean_object* v___x_1991_; 
v_toPure_1989_ = lean_ctor_get(v_toApplicative_1980_, 1);
lean_inc(v_toPure_1989_);
lean_dec_ref(v_toApplicative_1980_);
v___x_1990_ = lean_box(0);
v___x_1991_ = lean_apply_2(v_toPure_1989_, lean_box(0), v___x_1990_);
return v___x_1991_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_eraseGlobalRules___redArg___lam__5___boxed(lean_object* v_toApplicative_2011_, lean_object* v_checkExists_2012_, lean_object* v_rf_2013_, lean_object* v_val_2014_, lean_object* v___f_2015_, lean_object* v_inst_2016_, lean_object* v_inst_2017_, lean_object* v_anyErased_2018_){
_start:
{
uint8_t v_checkExists_boxed_2019_; uint8_t v_anyErased_boxed_2020_; lean_object* v_res_2021_; 
v_checkExists_boxed_2019_ = lean_unbox(v_checkExists_2012_);
v_anyErased_boxed_2020_ = lean_unbox(v_anyErased_2018_);
v_res_2021_ = lp_aesop_Aesop_Frontend_eraseGlobalRules___redArg___lam__5(v_toApplicative_2011_, v_checkExists_boxed_2019_, v_rf_2013_, v_val_2014_, v___f_2015_, v_inst_2016_, v_inst_2017_, v_anyErased_boxed_2020_);
return v_res_2021_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_eraseGlobalRules___redArg(lean_object* v_inst_2022_, lean_object* v_inst_2023_, lean_object* v_inst_2024_, lean_object* v_inst_2025_, lean_object* v_inst_2026_, lean_object* v_rsf_2027_, lean_object* v_rf_2028_, uint8_t v_checkExists_2029_){
_start:
{
lean_object* v_toApplicative_2030_; lean_object* v_toBind_2031_; lean_object* v___x_2032_; 
v_toApplicative_2030_ = lean_ctor_get(v_inst_2022_, 0);
v_toBind_2031_ = lean_ctor_get(v_inst_2022_, 1);
lean_inc(v_toBind_2031_);
v___x_2032_ = lp_aesop_Aesop_RuleSetNameFilter_matchedRuleSetNames(v_rsf_2027_);
if (lean_obj_tag(v___x_2032_) == 0)
{
lean_object* v___x_2033_; lean_object* v___f_2034_; lean_object* v___f_2035_; lean_object* v___f_2036_; lean_object* v___f_2037_; lean_object* v___x_2038_; lean_object* v___x_2039_; lean_object* v___x_2040_; 
lean_inc_ref_n(v_toApplicative_2030_, 2);
v___x_2033_ = lean_box(v_checkExists_2029_);
lean_inc_ref(v_inst_2023_);
lean_inc_ref_n(v_inst_2022_, 3);
lean_inc_ref(v_rf_2028_);
v___f_2034_ = lean_alloc_closure((void*)(lp_aesop_Aesop_Frontend_eraseGlobalRules___redArg___lam__0___boxed), 6, 5);
lean_closure_set(v___f_2034_, 0, v_toApplicative_2030_);
lean_closure_set(v___f_2034_, 1, v___x_2033_);
lean_closure_set(v___f_2034_, 2, v_rf_2028_);
lean_closure_set(v___f_2034_, 3, v_inst_2022_);
lean_closure_set(v___f_2034_, 4, v_inst_2023_);
lean_inc(v_inst_2024_);
v___f_2035_ = lean_alloc_closure((void*)(lp_aesop_Aesop_Frontend_eraseGlobalRules___redArg___lam__1___boxed), 9, 6);
lean_closure_set(v___f_2035_, 0, v_inst_2022_);
lean_closure_set(v___f_2035_, 1, v_inst_2023_);
lean_closure_set(v___f_2035_, 2, v_inst_2024_);
lean_closure_set(v___f_2035_, 3, v_inst_2025_);
lean_closure_set(v___f_2035_, 4, v_inst_2026_);
lean_closure_set(v___f_2035_, 5, v_rf_2028_);
v___f_2036_ = lean_alloc_closure((void*)(lp_aesop_Aesop_Frontend_eraseGlobalRules___redArg___lam__2___boxed), 4, 2);
lean_closure_set(v___f_2036_, 0, v_inst_2022_);
lean_closure_set(v___f_2036_, 1, v___f_2035_);
lean_inc(v_toBind_2031_);
v___f_2037_ = lean_alloc_closure((void*)(lp_aesop_Aesop_Frontend_eraseGlobalRules___redArg___lam__3), 6, 5);
lean_closure_set(v___f_2037_, 0, v_toApplicative_2030_);
lean_closure_set(v___f_2037_, 1, v_toBind_2031_);
lean_closure_set(v___f_2037_, 2, v___f_2034_);
lean_closure_set(v___f_2037_, 3, v_inst_2022_);
lean_closure_set(v___f_2037_, 4, v___f_2036_);
v___x_2038_ = ((lean_object*)(lp_aesop_Aesop_Frontend_getGlobalRuleSetData___redArg___closed__2));
v___x_2039_ = lean_apply_2(v_inst_2024_, lean_box(0), v___x_2038_);
v___x_2040_ = lean_apply_4(v_toBind_2031_, lean_box(0), lean_box(0), v___x_2039_, v___f_2037_);
return v___x_2040_;
}
else
{
lean_object* v_val_2041_; lean_object* v___x_2042_; lean_object* v___f_2043_; lean_object* v___x_2044_; lean_object* v___f_2045_; uint8_t v___x_2046_; lean_object* v___x_2047_; lean_object* v___x_2048_; uint8_t v___x_2049_; 
v_val_2041_ = lean_ctor_get(v___x_2032_, 0);
lean_inc_n(v_val_2041_, 2);
lean_dec_ref_known(v___x_2032_, 1);
v___x_2042_ = lean_box(v_checkExists_2029_);
v___f_2043_ = lean_alloc_closure((void*)(lp_aesop_Aesop_Frontend_eraseGlobalRules___redArg___lam__4___boxed), 2, 1);
lean_closure_set(v___f_2043_, 0, v___x_2042_);
v___x_2044_ = lean_box(v_checkExists_2029_);
lean_inc_ref(v_inst_2023_);
lean_inc_ref(v_inst_2022_);
lean_inc_ref(v_rf_2028_);
lean_inc_ref(v_toApplicative_2030_);
v___f_2045_ = lean_alloc_closure((void*)(lp_aesop_Aesop_Frontend_eraseGlobalRules___redArg___lam__5___boxed), 8, 7);
lean_closure_set(v___f_2045_, 0, v_toApplicative_2030_);
lean_closure_set(v___f_2045_, 1, v___x_2044_);
lean_closure_set(v___f_2045_, 2, v_rf_2028_);
lean_closure_set(v___f_2045_, 3, v_val_2041_);
lean_closure_set(v___f_2045_, 4, v___f_2043_);
lean_closure_set(v___f_2045_, 5, v_inst_2022_);
lean_closure_set(v___f_2045_, 6, v_inst_2023_);
v___x_2046_ = 0;
v___x_2047_ = lean_unsigned_to_nat(0u);
v___x_2048_ = lean_array_get_size(v_val_2041_);
v___x_2049_ = lean_nat_dec_lt(v___x_2047_, v___x_2048_);
if (v___x_2049_ == 0)
{
lean_object* v_toPure_2050_; lean_object* v___x_2051_; lean_object* v___x_2052_; lean_object* v___x_2053_; 
lean_inc_ref(v_toApplicative_2030_);
lean_dec(v_val_2041_);
lean_dec_ref(v_rf_2028_);
lean_dec_ref(v_inst_2026_);
lean_dec(v_inst_2025_);
lean_dec(v_inst_2024_);
lean_dec_ref(v_inst_2023_);
lean_dec_ref(v_inst_2022_);
v_toPure_2050_ = lean_ctor_get(v_toApplicative_2030_, 1);
lean_inc(v_toPure_2050_);
lean_dec_ref(v_toApplicative_2030_);
v___x_2051_ = lean_box(v___x_2046_);
v___x_2052_ = lean_apply_2(v_toPure_2050_, lean_box(0), v___x_2051_);
v___x_2053_ = lean_apply_4(v_toBind_2031_, lean_box(0), lean_box(0), v___x_2052_, v___f_2045_);
return v___x_2053_;
}
else
{
lean_object* v___x_2054_; uint8_t v___x_2055_; 
lean_inc_ref(v_inst_2022_);
v___x_2054_ = lean_alloc_closure((void*)(lp_aesop___private_Aesop_Frontend_Extension_0__Aesop_Frontend_eraseGlobalRules_go___boxed), 9, 7);
lean_closure_set(v___x_2054_, 0, lean_box(0));
lean_closure_set(v___x_2054_, 1, v_inst_2022_);
lean_closure_set(v___x_2054_, 2, v_inst_2023_);
lean_closure_set(v___x_2054_, 3, v_inst_2024_);
lean_closure_set(v___x_2054_, 4, v_inst_2025_);
lean_closure_set(v___x_2054_, 5, v_inst_2026_);
lean_closure_set(v___x_2054_, 6, v_rf_2028_);
v___x_2055_ = lean_nat_dec_le(v___x_2048_, v___x_2048_);
if (v___x_2055_ == 0)
{
if (v___x_2049_ == 0)
{
lean_object* v_toPure_2056_; lean_object* v___x_2057_; lean_object* v___x_2058_; lean_object* v___x_2059_; 
lean_inc_ref(v_toApplicative_2030_);
lean_dec_ref(v___x_2054_);
lean_dec(v_val_2041_);
lean_dec_ref(v_inst_2022_);
v_toPure_2056_ = lean_ctor_get(v_toApplicative_2030_, 1);
lean_inc(v_toPure_2056_);
lean_dec_ref(v_toApplicative_2030_);
v___x_2057_ = lean_box(v___x_2046_);
v___x_2058_ = lean_apply_2(v_toPure_2056_, lean_box(0), v___x_2057_);
v___x_2059_ = lean_apply_4(v_toBind_2031_, lean_box(0), lean_box(0), v___x_2058_, v___f_2045_);
return v___x_2059_;
}
else
{
size_t v___x_2060_; size_t v___x_2061_; lean_object* v___x_2062_; lean_object* v___x_2063_; lean_object* v___x_2064_; 
v___x_2060_ = ((size_t)0ULL);
v___x_2061_ = lean_usize_of_nat(v___x_2048_);
v___x_2062_ = lean_box(v___x_2046_);
v___x_2063_ = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), v_inst_2022_, v___x_2054_, v_val_2041_, v___x_2060_, v___x_2061_, v___x_2062_);
v___x_2064_ = lean_apply_4(v_toBind_2031_, lean_box(0), lean_box(0), v___x_2063_, v___f_2045_);
return v___x_2064_;
}
}
else
{
size_t v___x_2065_; size_t v___x_2066_; lean_object* v___x_2067_; lean_object* v___x_2068_; lean_object* v___x_2069_; 
v___x_2065_ = ((size_t)0ULL);
v___x_2066_ = lean_usize_of_nat(v___x_2048_);
v___x_2067_ = lean_box(v___x_2046_);
v___x_2068_ = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), v_inst_2022_, v___x_2054_, v_val_2041_, v___x_2065_, v___x_2066_, v___x_2067_);
v___x_2069_ = lean_apply_4(v_toBind_2031_, lean_box(0), lean_box(0), v___x_2068_, v___f_2045_);
return v___x_2069_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_eraseGlobalRules___redArg___boxed(lean_object* v_inst_2070_, lean_object* v_inst_2071_, lean_object* v_inst_2072_, lean_object* v_inst_2073_, lean_object* v_inst_2074_, lean_object* v_rsf_2075_, lean_object* v_rf_2076_, lean_object* v_checkExists_2077_){
_start:
{
uint8_t v_checkExists_boxed_2078_; lean_object* v_res_2079_; 
v_checkExists_boxed_2078_ = lean_unbox(v_checkExists_2077_);
v_res_2079_ = lp_aesop_Aesop_Frontend_eraseGlobalRules___redArg(v_inst_2070_, v_inst_2071_, v_inst_2072_, v_inst_2073_, v_inst_2074_, v_rsf_2075_, v_rf_2076_, v_checkExists_boxed_2078_);
return v_res_2079_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_eraseGlobalRules(lean_object* v_m_2080_, lean_object* v_inst_2081_, lean_object* v_inst_2082_, lean_object* v_inst_2083_, lean_object* v_inst_2084_, lean_object* v_inst_2085_, lean_object* v_rsf_2086_, lean_object* v_rf_2087_, uint8_t v_checkExists_2088_){
_start:
{
lean_object* v___x_2089_; 
v___x_2089_ = lp_aesop_Aesop_Frontend_eraseGlobalRules___redArg(v_inst_2081_, v_inst_2082_, v_inst_2083_, v_inst_2084_, v_inst_2085_, v_rsf_2086_, v_rf_2087_, v_checkExists_2088_);
return v___x_2089_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_eraseGlobalRules___boxed(lean_object* v_m_2090_, lean_object* v_inst_2091_, lean_object* v_inst_2092_, lean_object* v_inst_2093_, lean_object* v_inst_2094_, lean_object* v_inst_2095_, lean_object* v_rsf_2096_, lean_object* v_rf_2097_, lean_object* v_checkExists_2098_){
_start:
{
uint8_t v_checkExists_boxed_2099_; lean_object* v_res_2100_; 
v_checkExists_boxed_2099_ = lean_unbox(v_checkExists_2098_);
v_res_2100_ = lp_aesop_Aesop_Frontend_eraseGlobalRules(v_m_2090_, v_inst_2091_, v_inst_2092_, v_inst_2093_, v_inst_2094_, v_inst_2095_, v_rsf_2096_, v_rf_2097_, v_checkExists_boxed_2099_);
return v_res_2100_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_Frontend_Extension_Init(uint8_t builtin);
lean_object* runtime_initialize_Lean_Meta_Tactic_Simp_Simproc(uint8_t builtin);
lean_object* runtime_initialize_Lean_Meta_Tactic_Simp_Attr(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_aesop_Aesop_Frontend_Extension(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Frontend_Extension_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Meta_Tactic_Simp_Simproc(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Meta_Tactic_Simp_Attr(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = lp_aesop___private_Aesop_Frontend_Extension_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Extension_3818590508____hygCtx___hyg_2_();
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_aesop_Aesop_Frontend_Extension(uint8_t builtin) {
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
lean_object* initialize_aesop_Aesop_Frontend_Extension_Init(uint8_t builtin);
lean_object* initialize_Lean_Meta_Tactic_Simp_Simproc(uint8_t builtin);
lean_object* initialize_Lean_Meta_Tactic_Simp_Attr(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_aesop_Aesop_Frontend_Extension(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Frontend_Extension_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Meta_Tactic_Simp_Simproc(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Meta_Tactic_Simp_Attr(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Frontend_Extension(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_aesop_Aesop_Frontend_Extension(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_aesop_Aesop_Frontend_Extension(builtin);
}
#ifdef __cplusplus
}
#endif
