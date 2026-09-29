// Lean compiler output
// Module: Aesop.Frontend.Attribute
// Imports: public import Init public meta import Init public meta import Aesop.Frontend.Extension public meta import Aesop.Frontend.RuleExpr
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
extern lean_object* l_Lean_Elab_unsupportedSyntaxExceptionId;
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
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
extern lean_object* l_Lean_Meta_instInhabitedSimpTheorems_default;
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* l_Lean_MessageData_ofName(lean_object*);
lean_object* lean_st_ref_get(lean_object*);
lean_object* l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_RuleSetNameFilter_matchedRuleSetNames(lean_object*);
lean_object* lp_aesop_Aesop_getDeclaredRuleSets();
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
size_t lean_usize_of_nat(lean_object*);
uint8_t lean_usize_dec_eq(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
lean_object* lp_aesop_Aesop_GlobalRuleSet_erase(lean_object*, lean_object*);
uint64_t lean_uint64_shift_right(uint64_t, uint64_t);
uint64_t lean_uint64_xor(uint64_t, uint64_t);
size_t lean_usize_sub(size_t, size_t);
lean_object* l_Lean_Meta_getSimpExtension_x3f(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_Simp_getSimprocExtension_x3f(lean_object*);
lean_object* lean_io_error_to_string(lean_object*);
lean_object* l_Lean_MessageData_ofFormat(lean_object*);
extern lean_object* lp_aesop_Aesop_instInhabitedBaseRuleSet_default;
lean_object* l_Lean_ScopedEnvExtension_getState___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_Meta_Simp_instInhabitedSimprocs_default;
lean_object* l_Lean_ScopedEnvExtension_modifyState___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lean_st_ref_take(lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
size_t lean_usize_add(size_t, size_t);
size_t lean_array_size(lean_object*);
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* lean_array_uget(lean_object*, size_t);
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
lean_object* l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(lean_object*, uint8_t);
lean_object* lean_array_to_list(lean_object*);
lean_object* l_List_reverse___redArg(lean_object*);
lean_object* l_Lean_MessageData_ofList(lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Lean_Name_str___override(lean_object*, lean_object*);
lean_object* l_Lean_Name_num___override(lean_object*, lean_object*);
uint64_t l___private_Lean_Meta_Basic_0__Lean_Meta_Config_toKey(lean_object*);
lean_object* lean_st_mk_ref(lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_ElabM_Context_forAdditionalGlobalRules(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArgs(lean_object*);
lean_object* l_Lean_Syntax_TSepArray_getElems___redArg(lean_object*);
lean_object* lp_aesop_Aesop_Frontend_RuleExpr_elab(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_Frontend_RuleExpr_buildAdditionalGlobalRules(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Array_append___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Elab_Term_TermElabM_run___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_ScopedEnvExtension_addCore___redArg(lean_object*, lean_object*, lean_object*, uint8_t, lean_object*);
lean_object* lp_aesop_Aesop_GlobalRuleSetMember_name(lean_object*);
uint8_t lp_aesop_Aesop_GlobalRuleSet_contains(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr3(lean_object*, lean_object*, lean_object*);
extern lean_object* lp_aesop_Aesop_RuleSetNameFilter_all;
lean_object* l_Lean_registerBuiltinAttribute(lean_object*);
static const lean_string_object lp_aesop_Aesop_Frontend_Parser_Aesop_attr__rules_quot___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_Aesop_attr__rules_quot___closed__0 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_attr__rules_quot___closed__0_value;
static const lean_string_object lp_aesop_Aesop_Frontend_Parser_Aesop_attr__rules_quot___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_Aesop_attr__rules_quot___closed__1 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_attr__rules_quot___closed__1_value;
static const lean_string_object lp_aesop_Aesop_Frontend_Parser_Aesop_attr__rules_quot___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_Aesop_attr__rules_quot___closed__2 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_attr__rules_quot___closed__2_value;
static const lean_string_object lp_aesop_Aesop_Frontend_Parser_Aesop_attr__rules_quot___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "quot"};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_Aesop_attr__rules_quot___closed__3 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_attr__rules_quot___closed__3_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_Aesop_attr__rules_quot___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_attr__rules_quot___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_Aesop_attr__rules_quot___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_attr__rules_quot___closed__4_value_aux_0),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_attr__rules_quot___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_Aesop_attr__rules_quot___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_attr__rules_quot___closed__4_value_aux_1),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_attr__rules_quot___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_Aesop_attr__rules_quot___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_attr__rules_quot___closed__4_value_aux_2),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_attr__rules_quot___closed__3_value),LEAN_SCALAR_PTR_LITERAL(145, 163, 173, 41, 168, 168, 65, 81)}};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_Aesop_attr__rules_quot___closed__4 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_attr__rules_quot___closed__4_value;
static const lean_string_object lp_aesop_Aesop_Frontend_Parser_Aesop_attr__rules_quot___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "Aesop"};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_Aesop_attr__rules_quot___closed__5 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_attr__rules_quot___closed__5_value;
static const lean_string_object lp_aesop_Aesop_Frontend_Parser_Aesop_attr__rules_quot___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "attr_rules"};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_Aesop_attr__rules_quot___closed__6 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_attr__rules_quot___closed__6_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_Aesop_attr__rules_quot___closed__7_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_attr__rules_quot___closed__5_value),LEAN_SCALAR_PTR_LITERAL(97, 226, 101, 135, 78, 117, 164, 248)}};
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_Aesop_attr__rules_quot___closed__7_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_attr__rules_quot___closed__7_value_aux_0),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_attr__rules_quot___closed__6_value),LEAN_SCALAR_PTR_LITERAL(159, 152, 18, 77, 157, 98, 69, 225)}};
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_Aesop_attr__rules_quot___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_attr__rules_quot___closed__7_value_aux_1),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_attr__rules_quot___closed__3_value),LEAN_SCALAR_PTR_LITERAL(77, 76, 23, 41, 139, 161, 101, 245)}};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_Aesop_attr__rules_quot___closed__7 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_attr__rules_quot___closed__7_value;
static const lean_string_object lp_aesop_Aesop_Frontend_Parser_Aesop_attr__rules_quot___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_Aesop_attr__rules_quot___closed__8 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_attr__rules_quot___closed__8_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_Aesop_attr__rules_quot___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_attr__rules_quot___closed__8_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_Aesop_attr__rules_quot___closed__9 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_attr__rules_quot___closed__9_value;
static const lean_string_object lp_aesop_Aesop_Frontend_Parser_Aesop_attr__rules_quot___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "`(attr_rules| "};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_Aesop_attr__rules_quot___closed__10 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_attr__rules_quot___closed__10_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_Aesop_attr__rules_quot___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_attr__rules_quot___closed__10_value)}};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_Aesop_attr__rules_quot___closed__11 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_attr__rules_quot___closed__11_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_Aesop_attr__rules_quot___closed__12_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_attr__rules_quot___closed__5_value),LEAN_SCALAR_PTR_LITERAL(97, 226, 101, 135, 78, 117, 164, 248)}};
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_Aesop_attr__rules_quot___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_attr__rules_quot___closed__12_value_aux_0),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_attr__rules_quot___closed__6_value),LEAN_SCALAR_PTR_LITERAL(159, 152, 18, 77, 157, 98, 69, 225)}};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_Aesop_attr__rules_quot___closed__12 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_attr__rules_quot___closed__12_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_Aesop_attr__rules_quot___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_attr__rules_quot___closed__12_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_Aesop_attr__rules_quot___closed__13 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_attr__rules_quot___closed__13_value;
static const lean_string_object lp_aesop_Aesop_Frontend_Parser_Aesop_attr__rules_quot___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ")"};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_Aesop_attr__rules_quot___closed__14 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_attr__rules_quot___closed__14_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_Aesop_attr__rules_quot___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_attr__rules_quot___closed__14_value)}};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_Aesop_attr__rules_quot___closed__15 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_attr__rules_quot___closed__15_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_Aesop_attr__rules_quot___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_attr__rules_quot___closed__9_value),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_attr__rules_quot___closed__13_value),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_attr__rules_quot___closed__15_value)}};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_Aesop_attr__rules_quot___closed__16 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_attr__rules_quot___closed__16_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_Aesop_attr__rules_quot___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_attr__rules_quot___closed__9_value),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_attr__rules_quot___closed__11_value),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_attr__rules_quot___closed__16_value)}};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_Aesop_attr__rules_quot___closed__17 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_attr__rules_quot___closed__17_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_Aesop_attr__rules_quot___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_attr__rules_quot___closed__7_value),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_attr__rules_quot___closed__17_value)}};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_Aesop_attr__rules_quot___closed__18 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_attr__rules_quot___closed__18_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_Aesop_attr__rules_quot___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_attr__rules_quot___closed__4_value),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_attr__rules_quot___closed__18_value)}};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_Aesop_attr__rules_quot___closed__19 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_attr__rules_quot___closed__19_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_Frontend_Parser_Aesop_attr__rules_quot = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_attr__rules_quot___closed__19_value;
LEAN_EXPORT lean_object* lp_aesop_Lean_Parser_Category_Aesop_attr__rules;
static const lean_string_object lp_aesop_Aesop_Frontend_Parser_attr__rules___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "Frontend"};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_attr__rules___00__closed__0 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_attr__rules___00__closed__0_value;
static const lean_string_object lp_aesop_Aesop_Frontend_Parser_attr__rules___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "attr_rules_"};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_attr__rules___00__closed__1 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_attr__rules___00__closed__1_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_attr__rules___00__closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_attr__rules_quot___closed__5_value),LEAN_SCALAR_PTR_LITERAL(97, 226, 101, 135, 78, 117, 164, 248)}};
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_attr__rules___00__closed__2_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_attr__rules___00__closed__2_value_aux_0),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_attr__rules___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(212, 61, 86, 210, 90, 71, 197, 0)}};
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_attr__rules___00__closed__2_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_attr__rules___00__closed__2_value_aux_1),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_attr__rules_quot___closed__1_value),LEAN_SCALAR_PTR_LITERAL(13, 69, 189, 207, 58, 198, 186, 160)}};
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_attr__rules___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_attr__rules___00__closed__2_value_aux_2),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_attr__rules___00__closed__1_value),LEAN_SCALAR_PTR_LITERAL(117, 200, 0, 99, 172, 255, 207, 57)}};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_attr__rules___00__closed__2 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_attr__rules___00__closed__2_value;
static const lean_string_object lp_aesop_Aesop_Frontend_Parser_attr__rules___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "rule_expr"};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_attr__rules___00__closed__3 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_attr__rules___00__closed__3_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_attr__rules___00__closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_attr__rules_quot___closed__5_value),LEAN_SCALAR_PTR_LITERAL(97, 226, 101, 135, 78, 117, 164, 248)}};
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_attr__rules___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_attr__rules___00__closed__4_value_aux_0),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_attr__rules___00__closed__3_value),LEAN_SCALAR_PTR_LITERAL(99, 17, 168, 180, 164, 44, 144, 28)}};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_attr__rules___00__closed__4 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_attr__rules___00__closed__4_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_attr__rules___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_attr__rules___00__closed__4_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_attr__rules___00__closed__5 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_attr__rules___00__closed__5_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_attr__rules___00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_attr__rules___00__closed__2_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_attr__rules___00__closed__5_value)}};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_attr__rules___00__closed__6 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_attr__rules___00__closed__6_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_Frontend_Parser_attr__rules__ = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_attr__rules___00__closed__6_value;
static const lean_string_object lp_aesop_Aesop_Frontend_Parser_attr__rules_x5b___x5d___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "attr_rules[_]"};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_attr__rules_x5b___x5d___closed__0 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_attr__rules_x5b___x5d___closed__0_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_attr__rules_x5b___x5d___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_attr__rules_quot___closed__5_value),LEAN_SCALAR_PTR_LITERAL(97, 226, 101, 135, 78, 117, 164, 248)}};
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_attr__rules_x5b___x5d___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_attr__rules_x5b___x5d___closed__1_value_aux_0),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_attr__rules___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(212, 61, 86, 210, 90, 71, 197, 0)}};
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_attr__rules_x5b___x5d___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_attr__rules_x5b___x5d___closed__1_value_aux_1),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_attr__rules_quot___closed__1_value),LEAN_SCALAR_PTR_LITERAL(13, 69, 189, 207, 58, 198, 186, 160)}};
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_attr__rules_x5b___x5d___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_attr__rules_x5b___x5d___closed__1_value_aux_2),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_attr__rules_x5b___x5d___closed__0_value),LEAN_SCALAR_PTR_LITERAL(207, 6, 87, 12, 199, 110, 110, 168)}};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_attr__rules_x5b___x5d___closed__1 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_attr__rules_x5b___x5d___closed__1_value;
static const lean_string_object lp_aesop_Aesop_Frontend_Parser_attr__rules_x5b___x5d___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "["};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_attr__rules_x5b___x5d___closed__2 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_attr__rules_x5b___x5d___closed__2_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_attr__rules_x5b___x5d___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_attr__rules_x5b___x5d___closed__2_value)}};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_attr__rules_x5b___x5d___closed__3 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_attr__rules_x5b___x5d___closed__3_value;
static const lean_string_object lp_aesop_Aesop_Frontend_Parser_attr__rules_x5b___x5d___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ","};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_attr__rules_x5b___x5d___closed__4 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_attr__rules_x5b___x5d___closed__4_value;
static const lean_string_object lp_aesop_Aesop_Frontend_Parser_attr__rules_x5b___x5d___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = ", "};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_attr__rules_x5b___x5d___closed__5 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_attr__rules_x5b___x5d___closed__5_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_attr__rules_x5b___x5d___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_attr__rules_x5b___x5d___closed__5_value)}};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_attr__rules_x5b___x5d___closed__6 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_attr__rules_x5b___x5d___closed__6_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_attr__rules_x5b___x5d___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 8, .m_other = 3, .m_tag = 11}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_attr__rules___00__closed__5_value),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_attr__rules_x5b___x5d___closed__4_value),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_attr__rules_x5b___x5d___closed__6_value),LEAN_SCALAR_PTR_LITERAL(1, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_attr__rules_x5b___x5d___closed__7 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_attr__rules_x5b___x5d___closed__7_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_attr__rules_x5b___x5d___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_attr__rules_quot___closed__9_value),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_attr__rules_x5b___x5d___closed__3_value),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_attr__rules_x5b___x5d___closed__7_value)}};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_attr__rules_x5b___x5d___closed__8 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_attr__rules_x5b___x5d___closed__8_value;
static const lean_string_object lp_aesop_Aesop_Frontend_Parser_attr__rules_x5b___x5d___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "]"};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_attr__rules_x5b___x5d___closed__9 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_attr__rules_x5b___x5d___closed__9_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_attr__rules_x5b___x5d___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_attr__rules_x5b___x5d___closed__9_value)}};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_attr__rules_x5b___x5d___closed__10 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_attr__rules_x5b___x5d___closed__10_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_attr__rules_x5b___x5d___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_attr__rules_quot___closed__9_value),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_attr__rules_x5b___x5d___closed__8_value),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_attr__rules_x5b___x5d___closed__10_value)}};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_attr__rules_x5b___x5d___closed__11 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_attr__rules_x5b___x5d___closed__11_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_attr__rules_x5b___x5d___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_attr__rules_x5b___x5d___closed__1_value),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_attr__rules_x5b___x5d___closed__11_value)}};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_attr__rules_x5b___x5d___closed__12 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_attr__rules_x5b___x5d___closed__12_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_Frontend_Parser_attr__rules_x5b___x5d = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_attr__rules_x5b___x5d___closed__12_value;
static const lean_string_object lp_aesop_Aesop_Frontend_Parser_aesop___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "aesop"};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_aesop___closed__0 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_aesop___closed__0_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_aesop___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_attr__rules_quot___closed__5_value),LEAN_SCALAR_PTR_LITERAL(97, 226, 101, 135, 78, 117, 164, 248)}};
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_aesop___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_aesop___closed__1_value_aux_0),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_attr__rules___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(212, 61, 86, 210, 90, 71, 197, 0)}};
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_aesop___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_aesop___closed__1_value_aux_1),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_attr__rules_quot___closed__1_value),LEAN_SCALAR_PTR_LITERAL(13, 69, 189, 207, 58, 198, 186, 160)}};
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_aesop___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_aesop___closed__1_value_aux_2),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_aesop___closed__0_value),LEAN_SCALAR_PTR_LITERAL(203, 146, 252, 150, 212, 106, 105, 173)}};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_aesop___closed__1 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_aesop___closed__1_value;
static const lean_string_object lp_aesop_Aesop_Frontend_Parser_aesop___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "aesop "};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_aesop___closed__2 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_aesop___closed__2_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_aesop___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_aesop___closed__2_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_aesop___closed__3 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_aesop___closed__3_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_aesop___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_attr__rules_quot___closed__9_value),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_aesop___closed__3_value),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_attr__rules_quot___closed__13_value)}};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_aesop___closed__4 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_aesop___closed__4_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_aesop___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_aesop___closed__1_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_aesop___closed__4_value)}};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_aesop___closed__5 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_aesop___closed__5_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_Frontend_Parser_aesop = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_aesop___closed__5_value;
static const lean_array_object lp_aesop_Aesop_Frontend_instInhabitedAttrConfig_default___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_aesop_Aesop_Frontend_instInhabitedAttrConfig_default___closed__0 = (const lean_object*)&lp_aesop_Aesop_Frontend_instInhabitedAttrConfig_default___closed__0_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_Frontend_instInhabitedAttrConfig_default = (const lean_object*)&lp_aesop_Aesop_Frontend_instInhabitedAttrConfig_default___closed__0_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_Frontend_instInhabitedAttrConfig = (const lean_object*)&lp_aesop_Aesop_Frontend_instInhabitedAttrConfig_default___closed__0_value;
static lean_once_cell_t lp_aesop_Lean_Elab_throwUnsupportedSyntax___at___00Aesop_Frontend_AttrConfig_elab_spec__0___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Lean_Elab_throwUnsupportedSyntax___at___00Aesop_Frontend_AttrConfig_elab_spec__0___redArg___closed__0;
LEAN_EXPORT lean_object* lp_aesop_Lean_Elab_throwUnsupportedSyntax___at___00Aesop_Frontend_AttrConfig_elab_spec__0___redArg();
LEAN_EXPORT lean_object* lp_aesop_Lean_Elab_throwUnsupportedSyntax___at___00Aesop_Frontend_AttrConfig_elab_spec__0___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Elab_throwUnsupportedSyntax___at___00Aesop_Frontend_AttrConfig_elab_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Elab_throwUnsupportedSyntax___at___00Aesop_Frontend_AttrConfig_elab_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Frontend_AttrConfig_elab_spec__1(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Frontend_AttrConfig_elab_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_AttrConfig_elab(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_AttrConfig_elab___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___lam__0_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2_(lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___lam__0_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_modifyGetGlobalRuleSet___at___00__private_Aesop_Frontend_Extension_0__Aesop_Frontend_eraseGlobalRules_go___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0_spec__1_spec__3___redArg___lam__2(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_modifyGetGlobalRuleSet___at___00__private_Aesop_Frontend_Extension_0__Aesop_Frontend_eraseGlobalRules_go___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0_spec__1_spec__3___redArg___lam__2___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_modifyGetGlobalRuleSet___at___00__private_Aesop_Frontend_Extension_0__Aesop_Frontend_eraseGlobalRules_go___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0_spec__1_spec__3___redArg___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_modifyGetGlobalRuleSet___at___00__private_Aesop_Frontend_Extension_0__Aesop_Frontend_eraseGlobalRules_go___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0_spec__1_spec__3___redArg___lam__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__9_spec__12_spec__18___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__9_spec__12_spec__18___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__9_spec__12___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__9_spec__12___redArg___boxed(lean_object*, lean_object*);
static lean_once_cell_t lp_aesop_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0_spec__0_spec__1___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0_spec__0_spec__1___closed__0;
static lean_once_cell_t lp_aesop_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0_spec__0_spec__1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0_spec__0_spec__1___closed__1;
static lean_once_cell_t lp_aesop_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0_spec__0_spec__1___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0_spec__0_spec__1___closed__2;
static lean_once_cell_t lp_aesop_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0_spec__0_spec__1___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0_spec__0_spec__1___closed__3;
static lean_once_cell_t lp_aesop_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0_spec__0_spec__1___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0_spec__0_spec__1___closed__4;
static lean_once_cell_t lp_aesop_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0_spec__0_spec__1___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0_spec__0_spec__1___closed__5;
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0_spec__0_spec__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0_spec__0_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0_spec__0___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop_Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__9___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 27, .m_capacity = 27, .m_length = 26, .m_data = "internal error: expected '"};
static const lean_object* lp_aesop_Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__9___closed__0 = (const lean_object*)&lp_aesop_Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__9___closed__0_value;
static lean_once_cell_t lp_aesop_Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__9___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__9___closed__1;
static const lean_string_object lp_aesop_Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__9___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 34, .m_capacity = 34, .m_length = 33, .m_data = "' to be a declared simp extension"};
static const lean_object* lp_aesop_Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__9___closed__2 = (const lean_object*)&lp_aesop_Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__9___closed__2_value;
static lean_once_cell_t lp_aesop_Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__9___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__9___closed__3;
static const lean_string_object lp_aesop_Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__9___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "no such rule set: '"};
static const lean_object* lp_aesop_Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__9___closed__4 = (const lean_object*)&lp_aesop_Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__9___closed__4_value;
static lean_once_cell_t lp_aesop_Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__9___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__9___closed__5;
static const lean_string_object lp_aesop_Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__9___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 177, .m_capacity = 177, .m_length = 176, .m_data = "'\n  (Use 'declare_aesop_rule_set' to declare rule sets.\n   Declared rule sets are not visible in the current file; they only become visible once you import the declaring file.)"};
static const lean_object* lp_aesop_Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__9___closed__6 = (const lean_object*)&lp_aesop_Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__9___closed__6_value;
static lean_once_cell_t lp_aesop_Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__9___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__9___closed__7;
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__9(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__9___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_modifyGetGlobalRuleSet___at___00__private_Aesop_Frontend_Extension_0__Aesop_Frontend_eraseGlobalRules_go___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0_spec__1_spec__3___redArg___lam__1(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_modifyGetGlobalRuleSet___at___00__private_Aesop_Frontend_Extension_0__Aesop_Frontend_eraseGlobalRules_go___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0_spec__1_spec__3___redArg___lam__1___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_modifyGetGlobalRuleSet___at___00__private_Aesop_Frontend_Extension_0__Aesop_Frontend_eraseGlobalRules_go___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0_spec__1_spec__3___redArg___lam__3(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_modifyGetGlobalRuleSet___at___00__private_Aesop_Frontend_Extension_0__Aesop_Frontend_eraseGlobalRules_go___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0_spec__1_spec__3___redArg___lam__3___boxed(lean_object*, lean_object*);
static lean_once_cell_t lp_aesop_Lean_setEnv___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__12___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Lean_setEnv___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__12___redArg___closed__0;
static lean_once_cell_t lp_aesop_Lean_setEnv___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__12___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Lean_setEnv___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__12___redArg___closed__1;
static lean_once_cell_t lp_aesop_Lean_setEnv___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__12___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Lean_setEnv___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__12___redArg___closed__2;
LEAN_EXPORT lean_object* lp_aesop_Lean_setEnv___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__12___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_setEnv___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__12___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_modifyGetGlobalRuleSet___at___00__private_Aesop_Frontend_Extension_0__Aesop_Frontend_eraseGlobalRules_go___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0_spec__1_spec__3___redArg___lam__4(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_modifyGetGlobalRuleSet___at___00__private_Aesop_Frontend_Extension_0__Aesop_Frontend_eraseGlobalRules_go___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0_spec__1_spec__3___redArg___lam__4___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_aesop_Aesop_Frontend_modifyGetGlobalRuleSet___at___00__private_Aesop_Frontend_Extension_0__Aesop_Frontend_eraseGlobalRules_go___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0_spec__1_spec__3___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_Frontend_modifyGetGlobalRuleSet___at___00__private_Aesop_Frontend_Extension_0__Aesop_Frontend_eraseGlobalRules_go___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0_spec__1_spec__3___redArg___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_Frontend_modifyGetGlobalRuleSet___at___00__private_Aesop_Frontend_Extension_0__Aesop_Frontend_eraseGlobalRules_go___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0_spec__1_spec__3___redArg___closed__0 = (const lean_object*)&lp_aesop_Aesop_Frontend_modifyGetGlobalRuleSet___at___00__private_Aesop_Frontend_Extension_0__Aesop_Frontend_eraseGlobalRules_go___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0_spec__1_spec__3___redArg___closed__0_value;
static const lean_closure_object lp_aesop_Aesop_Frontend_modifyGetGlobalRuleSet___at___00__private_Aesop_Frontend_Extension_0__Aesop_Frontend_eraseGlobalRules_go___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0_spec__1_spec__3___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_Frontend_modifyGetGlobalRuleSet___at___00__private_Aesop_Frontend_Extension_0__Aesop_Frontend_eraseGlobalRules_go___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0_spec__1_spec__3___redArg___lam__1___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_Frontend_modifyGetGlobalRuleSet___at___00__private_Aesop_Frontend_Extension_0__Aesop_Frontend_eraseGlobalRules_go___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0_spec__1_spec__3___redArg___closed__1 = (const lean_object*)&lp_aesop_Aesop_Frontend_modifyGetGlobalRuleSet___at___00__private_Aesop_Frontend_Extension_0__Aesop_Frontend_eraseGlobalRules_go___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0_spec__1_spec__3___redArg___closed__1_value;
static const lean_closure_object lp_aesop_Aesop_Frontend_modifyGetGlobalRuleSet___at___00__private_Aesop_Frontend_Extension_0__Aesop_Frontend_eraseGlobalRules_go___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0_spec__1_spec__3___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_Frontend_modifyGetGlobalRuleSet___at___00__private_Aesop_Frontend_Extension_0__Aesop_Frontend_eraseGlobalRules_go___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0_spec__1_spec__3___redArg___lam__2___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_Frontend_modifyGetGlobalRuleSet___at___00__private_Aesop_Frontend_Extension_0__Aesop_Frontend_eraseGlobalRules_go___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0_spec__1_spec__3___redArg___closed__2 = (const lean_object*)&lp_aesop_Aesop_Frontend_modifyGetGlobalRuleSet___at___00__private_Aesop_Frontend_Extension_0__Aesop_Frontend_eraseGlobalRules_go___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0_spec__1_spec__3___redArg___closed__2_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_modifyGetGlobalRuleSet___at___00__private_Aesop_Frontend_Extension_0__Aesop_Frontend_eraseGlobalRules_go___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0_spec__1_spec__3___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_modifyGetGlobalRuleSet___at___00__private_Aesop_Frontend_Extension_0__Aesop_Frontend_eraseGlobalRules_go___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0_spec__1_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Frontend_Extension_0__Aesop_Frontend_eraseGlobalRules_go___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0_spec__1___lam__0(lean_object*, uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Frontend_Extension_0__Aesop_Frontend_eraseGlobalRules_go___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0_spec__1___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Frontend_Extension_0__Aesop_Frontend_eraseGlobalRules_go___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0_spec__1(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Frontend_Extension_0__Aesop_Frontend_eraseGlobalRules_go___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0_spec__6(lean_object*, lean_object*, size_t, size_t, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0_spec__2(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0_spec__3(lean_object*, lean_object*, size_t, size_t, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_List_mapTR_loop___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0_spec__5(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0_spec__4(uint8_t, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop_Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "'"};
static const lean_object* lp_aesop_Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0___closed__0 = (const lean_object*)&lp_aesop_Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0___closed__0_value;
static lean_once_cell_t lp_aesop_Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0___closed__1;
static const lean_string_object lp_aesop_Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 63, .m_capacity = 63, .m_length = 62, .m_data = "' is not registered (with the given features) in any rule set."};
static const lean_object* lp_aesop_Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0___closed__2 = (const lean_object*)&lp_aesop_Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0___closed__2_value;
static lean_once_cell_t lp_aesop_Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0___closed__3;
static const lean_string_object lp_aesop_Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 71, .m_capacity = 71, .m_length = 70, .m_data = "' is not registered (with the given features) in any of the rule sets "};
static const lean_object* lp_aesop_Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0___closed__4 = (const lean_object*)&lp_aesop_Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0___closed__4_value;
static lean_once_cell_t lp_aesop_Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0___closed__5;
static const lean_string_object lp_aesop_Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "."};
static const lean_object* lp_aesop_Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0___closed__6 = (const lean_object*)&lp_aesop_Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0___closed__6_value;
static lean_once_cell_t lp_aesop_Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0___closed__7;
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___lam__1___closed__0_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___lam__1___closed__0_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2_ = (const lean_object*)&lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___lam__1___closed__0_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__value;
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___lam__1_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2_(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___lam__1_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2____boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__1(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___lam__2___closed__0_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___lam__2___closed__0_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2_ = (const lean_object*)&lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___lam__2___closed__0_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__value;
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___lam__2_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2_(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___lam__2_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2____boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_ScopedEnvExtension_add___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__10___redArg(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_ScopedEnvExtension_add___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__10___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Array_idxOfAux___at___00Array_finIdxOf_x3f___at___00Lean_PersistentHashMap_eraseAux___at___00Lean_PersistentHashMap_erase___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__11_spec__15_spec__21_spec__23(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Array_idxOfAux___at___00Array_finIdxOf_x3f___at___00Lean_PersistentHashMap_eraseAux___at___00Lean_PersistentHashMap_erase___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__11_spec__15_spec__21_spec__23___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Array_finIdxOf_x3f___at___00Lean_PersistentHashMap_eraseAux___at___00Lean_PersistentHashMap_erase___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__11_spec__15_spec__21(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Array_finIdxOf_x3f___at___00Lean_PersistentHashMap_eraseAux___at___00Lean_PersistentHashMap_erase___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__11_spec__15_spec__21___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_eraseAux___at___00Lean_PersistentHashMap_erase___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__11_spec__15___redArg(lean_object*, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_eraseAux___at___00Lean_PersistentHashMap_erase___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__11_spec__15___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_erase___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__11___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_erase___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__11___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__13___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__13___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__13(lean_object*, uint8_t, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__13___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_getGlobalRuleSetFromData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__14___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_getGlobalRuleSetFromData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__14___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop_Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "aesop: rule '"};
static const lean_object* lp_aesop_Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2___closed__0 = (const lean_object*)&lp_aesop_Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2___closed__0_value;
static lean_once_cell_t lp_aesop_Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2___closed__1;
static const lean_string_object lp_aesop_Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 38, .m_capacity = 38, .m_length = 37, .m_data = "' is already registered in rule set '"};
static const lean_object* lp_aesop_Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2___closed__2 = (const lean_object*)&lp_aesop_Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2___closed__2_value;
static lean_once_cell_t lp_aesop_Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2___closed__3;
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__3(lean_object*, uint8_t, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__4(uint8_t, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___lam__3___closed__0_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___lam__3___closed__0_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2_ = (const lean_object*)&lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___lam__3___closed__0_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__value;
static const lean_ctor_object lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___lam__3___closed__1_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*0 + 24, .m_other = 0, .m_tag = 0}, .m_objs = {LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 1, 1, 0),LEAN_SCALAR_PTR_LITERAL(1, 1, 0, 1, 1, 1, 2, 1),LEAN_SCALAR_PTR_LITERAL(1, 1, 1, 0, 0, 0, 0, 0)}};
static const lean_object* lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___lam__3___closed__1_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2_ = (const lean_object*)&lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___lam__3___closed__1_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__value;
static lean_once_cell_t lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___lam__3___closed__2_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static uint64_t lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___lam__3___closed__2_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2_;
static lean_once_cell_t lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___lam__3___closed__3_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___lam__3___closed__3_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2_;
static lean_once_cell_t lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___lam__3___closed__4_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___lam__3___closed__4_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2_;
static lean_once_cell_t lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___lam__3___closed__5_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___lam__3___closed__5_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2_;
static lean_once_cell_t lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___lam__3___closed__6_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___lam__3___closed__6_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2_;
static lean_once_cell_t lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___lam__3___closed__7_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___lam__3___closed__7_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2_;
static lean_once_cell_t lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___lam__3___closed__8_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___lam__3___closed__8_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2_;
static lean_once_cell_t lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___lam__3___closed__9_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___lam__3___closed__9_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2_;
static lean_once_cell_t lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___lam__3___closed__10_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___lam__3___closed__10_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2_;
static lean_once_cell_t lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___lam__3___closed__11_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___lam__3___closed__11_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2_;
static const lean_ctor_object lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___lam__3___closed__12_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*7 + 0, .m_other = 7, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(1) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(1) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___lam__3___closed__12_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2_ = (const lean_object*)&lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___lam__3___closed__12_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__value;
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___lam__3_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2_(lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___lam__3_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2____boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___closed__0_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___lam__0_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2____boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___closed__0_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2_ = (const lean_object*)&lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___closed__0_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__value;
static const lean_closure_object lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___closed__1_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___lam__1_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2____boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___closed__1_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2_ = (const lean_object*)&lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___closed__1_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__value;
static const lean_closure_object lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___closed__2_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___lam__3_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2____boxed, .m_arity = 7, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___closed__0_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__value)} };
static const lean_object* lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___closed__2_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2_ = (const lean_object*)&lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___closed__2_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__value;
static const lean_string_object lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___closed__3_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "_private"};
static const lean_object* lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___closed__3_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2_ = (const lean_object*)&lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___closed__3_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__value;
static const lean_ctor_object lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___closed__4_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___closed__3_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(103, 214, 75, 80, 34, 198, 193, 153)}};
static const lean_object* lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___closed__4_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2_ = (const lean_object*)&lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___closed__4_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__value;
static const lean_ctor_object lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___closed__5_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___closed__4_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__value),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_attr__rules_quot___closed__5_value),LEAN_SCALAR_PTR_LITERAL(213, 96, 250, 13, 195, 1, 48, 100)}};
static const lean_object* lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___closed__5_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2_ = (const lean_object*)&lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___closed__5_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__value;
static const lean_ctor_object lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___closed__6_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___closed__5_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__value),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_attr__rules___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(240, 66, 212, 89, 115, 115, 135, 73)}};
static const lean_object* lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___closed__6_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2_ = (const lean_object*)&lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___closed__6_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__value;
static const lean_string_object lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___closed__7_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "Attribute"};
static const lean_object* lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___closed__7_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2_ = (const lean_object*)&lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___closed__7_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__value;
static const lean_ctor_object lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___closed__8_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___closed__6_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__value),((lean_object*)&lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___closed__7_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(27, 57, 220, 90, 47, 197, 196, 152)}};
static const lean_object* lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___closed__8_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2_ = (const lean_object*)&lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___closed__8_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__value;
static const lean_ctor_object lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___closed__9_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___closed__8_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__value),((lean_object*)(((size_t)(0) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(166, 160, 201, 237, 85, 189, 198, 36)}};
static const lean_object* lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___closed__9_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2_ = (const lean_object*)&lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___closed__9_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__value;
static const lean_ctor_object lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___closed__10_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___closed__9_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__value),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_attr__rules_quot___closed__5_value),LEAN_SCALAR_PTR_LITERAL(128, 182, 134, 104, 156, 37, 254, 190)}};
static const lean_object* lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___closed__10_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2_ = (const lean_object*)&lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___closed__10_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__value;
static const lean_ctor_object lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___closed__11_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___closed__10_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__value),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_attr__rules___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(201, 184, 135, 207, 190, 138, 242, 78)}};
static const lean_object* lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___closed__11_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2_ = (const lean_object*)&lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___closed__11_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__value;
static const lean_string_object lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___closed__12_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "initFn"};
static const lean_object* lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___closed__12_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2_ = (const lean_object*)&lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___closed__12_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__value;
static const lean_ctor_object lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___closed__13_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___closed__11_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__value),((lean_object*)&lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___closed__12_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(208, 133, 60, 201, 233, 5, 179, 223)}};
static const lean_object* lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___closed__13_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2_ = (const lean_object*)&lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___closed__13_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__value;
static const lean_string_object lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___closed__14_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "_@"};
static const lean_object* lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___closed__14_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2_ = (const lean_object*)&lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___closed__14_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__value;
static const lean_ctor_object lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___closed__15_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___closed__13_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__value),((lean_object*)&lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___closed__14_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(81, 200, 210, 79, 165, 254, 115, 134)}};
static const lean_object* lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___closed__15_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2_ = (const lean_object*)&lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___closed__15_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__value;
static const lean_ctor_object lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___closed__16_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___closed__15_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__value),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_Aesop_attr__rules_quot___closed__5_value),LEAN_SCALAR_PTR_LITERAL(67, 120, 68, 251, 187, 87, 121, 161)}};
static const lean_object* lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___closed__16_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2_ = (const lean_object*)&lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___closed__16_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__value;
static const lean_ctor_object lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___closed__17_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___closed__16_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__value),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_attr__rules___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(158, 245, 249, 147, 204, 87, 35, 136)}};
static const lean_object* lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___closed__17_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2_ = (const lean_object*)&lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___closed__17_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__value;
static const lean_ctor_object lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___closed__18_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___closed__17_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__value),((lean_object*)&lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___closed__7_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(21, 224, 197, 64, 50, 194, 232, 153)}};
static const lean_object* lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___closed__18_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2_ = (const lean_object*)&lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___closed__18_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__value;
static lean_once_cell_t lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___closed__19_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___closed__19_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2_;
static const lean_string_object lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___closed__20_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "_hygCtx"};
static const lean_object* lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___closed__20_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2_ = (const lean_object*)&lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___closed__20_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__value;
static lean_once_cell_t lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___closed__21_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___closed__21_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2_;
static const lean_string_object lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___closed__22_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "_hyg"};
static const lean_object* lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___closed__22_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2_ = (const lean_object*)&lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___closed__22_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__value;
static lean_once_cell_t lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___closed__23_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___closed__23_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2_;
static lean_once_cell_t lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___closed__24_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___closed__24_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2_;
static const lean_ctor_object lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___closed__25_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_aesop___closed__0_value),LEAN_SCALAR_PTR_LITERAL(29, 147, 5, 234, 46, 7, 240, 183)}};
static const lean_object* lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___closed__25_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2_ = (const lean_object*)&lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___closed__25_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__value;
static const lean_string_object lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___closed__26_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 41, .m_capacity = 41, .m_length = 40, .m_data = "Register a declaration as an Aesop rule."};
static const lean_object* lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___closed__26_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2_ = (const lean_object*)&lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___closed__26_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__value;
static lean_once_cell_t lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___closed__27_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___closed__27_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2_;
static lean_once_cell_t lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___closed__28_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___closed__28_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2_;
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2_();
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_ScopedEnvExtension_add___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__10(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_ScopedEnvExtension_add___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__10___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_setEnv___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__12(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_setEnv___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__12___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_getGlobalRuleSetFromData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__14(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_getGlobalRuleSetFromData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__14___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_erase___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__11(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_erase___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__11___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_modifyGetGlobalRuleSet___at___00__private_Aesop_Frontend_Extension_0__Aesop_Frontend_eraseGlobalRules_go___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0_spec__1_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_modifyGetGlobalRuleSet___at___00__private_Aesop_Frontend_Extension_0__Aesop_Frontend_eraseGlobalRules_go___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0_spec__1_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__9_spec__12(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__9_spec__12___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_eraseAux___at___00Lean_PersistentHashMap_erase___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__11_spec__15(lean_object*, lean_object*, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_eraseAux___at___00Lean_PersistentHashMap_erase___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__11_spec__15___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__9_spec__12_spec__18(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__9_spec__12_spec__18___boxed(lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_aesop_Lean_Parser_Category_Aesop_attr__rules(void){
_start:
{
lean_object* v___x_48_; 
v___x_48_ = lean_box(0);
return v___x_48_;
}
}
static lean_object* _init_lp_aesop_Lean_Elab_throwUnsupportedSyntax___at___00Aesop_Frontend_AttrConfig_elab_spec__0___redArg___closed__0(void){
_start:
{
lean_object* v___x_125_; lean_object* v___x_126_; lean_object* v___x_127_; 
v___x_125_ = lean_box(0);
v___x_126_ = l_Lean_Elab_unsupportedSyntaxExceptionId;
v___x_127_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_127_, 0, v___x_126_);
lean_ctor_set(v___x_127_, 1, v___x_125_);
return v___x_127_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Elab_throwUnsupportedSyntax___at___00Aesop_Frontend_AttrConfig_elab_spec__0___redArg(){
_start:
{
lean_object* v___x_129_; lean_object* v___x_130_; 
v___x_129_ = lean_obj_once(&lp_aesop_Lean_Elab_throwUnsupportedSyntax___at___00Aesop_Frontend_AttrConfig_elab_spec__0___redArg___closed__0, &lp_aesop_Lean_Elab_throwUnsupportedSyntax___at___00Aesop_Frontend_AttrConfig_elab_spec__0___redArg___closed__0_once, _init_lp_aesop_Lean_Elab_throwUnsupportedSyntax___at___00Aesop_Frontend_AttrConfig_elab_spec__0___redArg___closed__0);
v___x_130_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_130_, 0, v___x_129_);
return v___x_130_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Elab_throwUnsupportedSyntax___at___00Aesop_Frontend_AttrConfig_elab_spec__0___redArg___boxed(lean_object* v___y_131_){
_start:
{
lean_object* v_res_132_; 
v_res_132_ = lp_aesop_Lean_Elab_throwUnsupportedSyntax___at___00Aesop_Frontend_AttrConfig_elab_spec__0___redArg();
return v_res_132_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Elab_throwUnsupportedSyntax___at___00Aesop_Frontend_AttrConfig_elab_spec__0(lean_object* v_00_u03b1_133_, lean_object* v___y_134_, lean_object* v___y_135_, lean_object* v___y_136_, lean_object* v___y_137_, lean_object* v___y_138_, lean_object* v___y_139_){
_start:
{
lean_object* v___x_141_; 
v___x_141_ = lp_aesop_Lean_Elab_throwUnsupportedSyntax___at___00Aesop_Frontend_AttrConfig_elab_spec__0___redArg();
return v___x_141_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Elab_throwUnsupportedSyntax___at___00Aesop_Frontend_AttrConfig_elab_spec__0___boxed(lean_object* v_00_u03b1_142_, lean_object* v___y_143_, lean_object* v___y_144_, lean_object* v___y_145_, lean_object* v___y_146_, lean_object* v___y_147_, lean_object* v___y_148_, lean_object* v___y_149_){
_start:
{
lean_object* v_res_150_; 
v_res_150_ = lp_aesop_Lean_Elab_throwUnsupportedSyntax___at___00Aesop_Frontend_AttrConfig_elab_spec__0(v_00_u03b1_142_, v___y_143_, v___y_144_, v___y_145_, v___y_146_, v___y_147_, v___y_148_);
lean_dec(v___y_148_);
lean_dec_ref(v___y_147_);
lean_dec(v___y_146_);
lean_dec_ref(v___y_145_);
lean_dec(v___y_144_);
lean_dec_ref(v___y_143_);
return v_res_150_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Frontend_AttrConfig_elab_spec__1(lean_object* v_a_151_, size_t v_sz_152_, size_t v_i_153_, lean_object* v_bs_154_, lean_object* v___y_155_, lean_object* v___y_156_, lean_object* v___y_157_, lean_object* v___y_158_, lean_object* v___y_159_, lean_object* v___y_160_){
_start:
{
uint8_t v___x_162_; 
v___x_162_ = lean_usize_dec_lt(v_i_153_, v_sz_152_);
if (v___x_162_ == 0)
{
lean_object* v___x_163_; 
v___x_163_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_163_, 0, v_bs_154_);
return v___x_163_;
}
else
{
lean_object* v_v_164_; lean_object* v___x_165_; 
v_v_164_ = lean_array_uget_borrowed(v_bs_154_, v_i_153_);
lean_inc(v_v_164_);
v___x_165_ = lp_aesop_Aesop_Frontend_RuleExpr_elab(v_v_164_, v_a_151_, v___y_155_, v___y_156_, v___y_157_, v___y_158_, v___y_159_, v___y_160_);
if (lean_obj_tag(v___x_165_) == 0)
{
lean_object* v_a_166_; lean_object* v___x_167_; lean_object* v_bs_x27_168_; size_t v___x_169_; size_t v___x_170_; lean_object* v___x_171_; 
v_a_166_ = lean_ctor_get(v___x_165_, 0);
lean_inc(v_a_166_);
lean_dec_ref_known(v___x_165_, 1);
v___x_167_ = lean_unsigned_to_nat(0u);
v_bs_x27_168_ = lean_array_uset(v_bs_154_, v_i_153_, v___x_167_);
v___x_169_ = ((size_t)1ULL);
v___x_170_ = lean_usize_add(v_i_153_, v___x_169_);
v___x_171_ = lean_array_uset(v_bs_x27_168_, v_i_153_, v_a_166_);
v_i_153_ = v___x_170_;
v_bs_154_ = v___x_171_;
goto _start;
}
else
{
lean_object* v_a_173_; lean_object* v___x_175_; uint8_t v_isShared_176_; uint8_t v_isSharedCheck_180_; 
lean_dec_ref(v_bs_154_);
v_a_173_ = lean_ctor_get(v___x_165_, 0);
v_isSharedCheck_180_ = !lean_is_exclusive(v___x_165_);
if (v_isSharedCheck_180_ == 0)
{
v___x_175_ = v___x_165_;
v_isShared_176_ = v_isSharedCheck_180_;
goto v_resetjp_174_;
}
else
{
lean_inc(v_a_173_);
lean_dec(v___x_165_);
v___x_175_ = lean_box(0);
v_isShared_176_ = v_isSharedCheck_180_;
goto v_resetjp_174_;
}
v_resetjp_174_:
{
lean_object* v___x_178_; 
if (v_isShared_176_ == 0)
{
v___x_178_ = v___x_175_;
goto v_reusejp_177_;
}
else
{
lean_object* v_reuseFailAlloc_179_; 
v_reuseFailAlloc_179_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_179_, 0, v_a_173_);
v___x_178_ = v_reuseFailAlloc_179_;
goto v_reusejp_177_;
}
v_reusejp_177_:
{
return v___x_178_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Frontend_AttrConfig_elab_spec__1___boxed(lean_object* v_a_181_, lean_object* v_sz_182_, lean_object* v_i_183_, lean_object* v_bs_184_, lean_object* v___y_185_, lean_object* v___y_186_, lean_object* v___y_187_, lean_object* v___y_188_, lean_object* v___y_189_, lean_object* v___y_190_, lean_object* v___y_191_){
_start:
{
size_t v_sz_boxed_192_; size_t v_i_boxed_193_; lean_object* v_res_194_; 
v_sz_boxed_192_ = lean_unbox_usize(v_sz_182_);
lean_dec(v_sz_182_);
v_i_boxed_193_ = lean_unbox_usize(v_i_183_);
lean_dec(v_i_183_);
v_res_194_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Frontend_AttrConfig_elab_spec__1(v_a_181_, v_sz_boxed_192_, v_i_boxed_193_, v_bs_184_, v___y_185_, v___y_186_, v___y_187_, v___y_188_, v___y_189_, v___y_190_);
lean_dec(v___y_190_);
lean_dec_ref(v___y_189_);
lean_dec(v___y_188_);
lean_dec_ref(v___y_187_);
lean_dec(v___y_186_);
lean_dec_ref(v___y_185_);
lean_dec_ref(v_a_181_);
return v_res_194_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_AttrConfig_elab(lean_object* v_stx_195_, lean_object* v_a_196_, lean_object* v_a_197_, lean_object* v_a_198_, lean_object* v_a_199_, lean_object* v_a_200_, lean_object* v_a_201_){
_start:
{
lean_object* v_fileName_203_; lean_object* v_fileMap_204_; lean_object* v_options_205_; lean_object* v_currRecDepth_206_; lean_object* v_maxRecDepth_207_; lean_object* v_ref_208_; lean_object* v_currNamespace_209_; lean_object* v_openDecls_210_; lean_object* v_initHeartbeats_211_; lean_object* v_maxHeartbeats_212_; lean_object* v_quotContext_213_; lean_object* v_currMacroScope_214_; uint8_t v_diag_215_; lean_object* v_cancelTk_x3f_216_; uint8_t v_suppressElabErrors_217_; lean_object* v_inheritedTraceOptions_218_; lean_object* v___x_219_; uint8_t v___x_220_; 
v_fileName_203_ = lean_ctor_get(v_a_200_, 0);
v_fileMap_204_ = lean_ctor_get(v_a_200_, 1);
v_options_205_ = lean_ctor_get(v_a_200_, 2);
v_currRecDepth_206_ = lean_ctor_get(v_a_200_, 3);
v_maxRecDepth_207_ = lean_ctor_get(v_a_200_, 4);
v_ref_208_ = lean_ctor_get(v_a_200_, 5);
v_currNamespace_209_ = lean_ctor_get(v_a_200_, 6);
v_openDecls_210_ = lean_ctor_get(v_a_200_, 7);
v_initHeartbeats_211_ = lean_ctor_get(v_a_200_, 8);
v_maxHeartbeats_212_ = lean_ctor_get(v_a_200_, 9);
v_quotContext_213_ = lean_ctor_get(v_a_200_, 10);
v_currMacroScope_214_ = lean_ctor_get(v_a_200_, 11);
v_diag_215_ = lean_ctor_get_uint8(v_a_200_, sizeof(void*)*14);
v_cancelTk_x3f_216_ = lean_ctor_get(v_a_200_, 12);
v_suppressElabErrors_217_ = lean_ctor_get_uint8(v_a_200_, sizeof(void*)*14 + 1);
v_inheritedTraceOptions_218_ = lean_ctor_get(v_a_200_, 13);
v___x_219_ = ((lean_object*)(lp_aesop_Aesop_Frontend_Parser_aesop___closed__1));
lean_inc(v_stx_195_);
v___x_220_ = l_Lean_Syntax_isOfKind(v_stx_195_, v___x_219_);
if (v___x_220_ == 0)
{
lean_object* v___x_221_; 
lean_dec(v_stx_195_);
v___x_221_ = lp_aesop_Lean_Elab_throwUnsupportedSyntax___at___00Aesop_Frontend_AttrConfig_elab_spec__0___redArg();
return v___x_221_;
}
else
{
lean_object* v_ref_222_; lean_object* v___x_223_; lean_object* v___x_224_; lean_object* v___x_225_; lean_object* v___x_226_; uint8_t v___x_227_; 
v_ref_222_ = l_Lean_replaceRef(v_stx_195_, v_ref_208_);
lean_inc_ref(v_inheritedTraceOptions_218_);
lean_inc(v_cancelTk_x3f_216_);
lean_inc(v_currMacroScope_214_);
lean_inc(v_quotContext_213_);
lean_inc(v_maxHeartbeats_212_);
lean_inc(v_initHeartbeats_211_);
lean_inc(v_openDecls_210_);
lean_inc(v_currNamespace_209_);
lean_inc(v_maxRecDepth_207_);
lean_inc(v_currRecDepth_206_);
lean_inc_ref(v_options_205_);
lean_inc_ref(v_fileMap_204_);
lean_inc_ref(v_fileName_203_);
v___x_223_ = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(v___x_223_, 0, v_fileName_203_);
lean_ctor_set(v___x_223_, 1, v_fileMap_204_);
lean_ctor_set(v___x_223_, 2, v_options_205_);
lean_ctor_set(v___x_223_, 3, v_currRecDepth_206_);
lean_ctor_set(v___x_223_, 4, v_maxRecDepth_207_);
lean_ctor_set(v___x_223_, 5, v_ref_222_);
lean_ctor_set(v___x_223_, 6, v_currNamespace_209_);
lean_ctor_set(v___x_223_, 7, v_openDecls_210_);
lean_ctor_set(v___x_223_, 8, v_initHeartbeats_211_);
lean_ctor_set(v___x_223_, 9, v_maxHeartbeats_212_);
lean_ctor_set(v___x_223_, 10, v_quotContext_213_);
lean_ctor_set(v___x_223_, 11, v_currMacroScope_214_);
lean_ctor_set(v___x_223_, 12, v_cancelTk_x3f_216_);
lean_ctor_set(v___x_223_, 13, v_inheritedTraceOptions_218_);
lean_ctor_set_uint8(v___x_223_, sizeof(void*)*14, v_diag_215_);
lean_ctor_set_uint8(v___x_223_, sizeof(void*)*14 + 1, v_suppressElabErrors_217_);
v___x_224_ = lean_unsigned_to_nat(1u);
v___x_225_ = l_Lean_Syntax_getArg(v_stx_195_, v___x_224_);
lean_dec(v_stx_195_);
v___x_226_ = ((lean_object*)(lp_aesop_Aesop_Frontend_Parser_attr__rules___00__closed__2));
lean_inc(v___x_225_);
v___x_227_ = l_Lean_Syntax_isOfKind(v___x_225_, v___x_226_);
if (v___x_227_ == 0)
{
lean_object* v___x_228_; uint8_t v___x_229_; 
v___x_228_ = ((lean_object*)(lp_aesop_Aesop_Frontend_Parser_attr__rules_x5b___x5d___closed__1));
lean_inc(v___x_225_);
v___x_229_ = l_Lean_Syntax_isOfKind(v___x_225_, v___x_228_);
if (v___x_229_ == 0)
{
lean_object* v___x_230_; 
lean_dec(v___x_225_);
lean_dec_ref_known(v___x_223_, 14);
v___x_230_ = lp_aesop_Lean_Elab_throwUnsupportedSyntax___at___00Aesop_Frontend_AttrConfig_elab_spec__0___redArg();
return v___x_230_;
}
else
{
lean_object* v___x_231_; 
v___x_231_ = lp_aesop_Aesop_ElabM_Context_forAdditionalGlobalRules(v_a_198_, v_a_199_, v___x_223_, v_a_201_);
if (lean_obj_tag(v___x_231_) == 0)
{
lean_object* v_a_232_; lean_object* v___x_233_; lean_object* v_es_234_; lean_object* v___x_235_; size_t v_sz_236_; size_t v___x_237_; lean_object* v___x_238_; 
v_a_232_ = lean_ctor_get(v___x_231_, 0);
lean_inc(v_a_232_);
lean_dec_ref_known(v___x_231_, 1);
v___x_233_ = l_Lean_Syntax_getArg(v___x_225_, v___x_224_);
lean_dec(v___x_225_);
v_es_234_ = l_Lean_Syntax_getArgs(v___x_233_);
lean_dec(v___x_233_);
v___x_235_ = l_Lean_Syntax_TSepArray_getElems___redArg(v_es_234_);
lean_dec_ref(v_es_234_);
v_sz_236_ = lean_array_size(v___x_235_);
v___x_237_ = ((size_t)0ULL);
v___x_238_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Frontend_AttrConfig_elab_spec__1(v_a_232_, v_sz_236_, v___x_237_, v___x_235_, v_a_196_, v_a_197_, v_a_198_, v_a_199_, v___x_223_, v_a_201_);
lean_dec_ref_known(v___x_223_, 14);
lean_dec(v_a_232_);
if (lean_obj_tag(v___x_238_) == 0)
{
lean_object* v_a_239_; lean_object* v___x_241_; uint8_t v_isShared_242_; uint8_t v_isSharedCheck_246_; 
v_a_239_ = lean_ctor_get(v___x_238_, 0);
v_isSharedCheck_246_ = !lean_is_exclusive(v___x_238_);
if (v_isSharedCheck_246_ == 0)
{
v___x_241_ = v___x_238_;
v_isShared_242_ = v_isSharedCheck_246_;
goto v_resetjp_240_;
}
else
{
lean_inc(v_a_239_);
lean_dec(v___x_238_);
v___x_241_ = lean_box(0);
v_isShared_242_ = v_isSharedCheck_246_;
goto v_resetjp_240_;
}
v_resetjp_240_:
{
lean_object* v___x_244_; 
if (v_isShared_242_ == 0)
{
v___x_244_ = v___x_241_;
goto v_reusejp_243_;
}
else
{
lean_object* v_reuseFailAlloc_245_; 
v_reuseFailAlloc_245_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_245_, 0, v_a_239_);
v___x_244_ = v_reuseFailAlloc_245_;
goto v_reusejp_243_;
}
v_reusejp_243_:
{
return v___x_244_;
}
}
}
else
{
lean_object* v_a_247_; lean_object* v___x_249_; uint8_t v_isShared_250_; uint8_t v_isSharedCheck_254_; 
v_a_247_ = lean_ctor_get(v___x_238_, 0);
v_isSharedCheck_254_ = !lean_is_exclusive(v___x_238_);
if (v_isSharedCheck_254_ == 0)
{
v___x_249_ = v___x_238_;
v_isShared_250_ = v_isSharedCheck_254_;
goto v_resetjp_248_;
}
else
{
lean_inc(v_a_247_);
lean_dec(v___x_238_);
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
else
{
lean_object* v_a_255_; lean_object* v___x_257_; uint8_t v_isShared_258_; uint8_t v_isSharedCheck_262_; 
lean_dec(v___x_225_);
lean_dec_ref_known(v___x_223_, 14);
v_a_255_ = lean_ctor_get(v___x_231_, 0);
v_isSharedCheck_262_ = !lean_is_exclusive(v___x_231_);
if (v_isSharedCheck_262_ == 0)
{
v___x_257_ = v___x_231_;
v_isShared_258_ = v_isSharedCheck_262_;
goto v_resetjp_256_;
}
else
{
lean_inc(v_a_255_);
lean_dec(v___x_231_);
v___x_257_ = lean_box(0);
v_isShared_258_ = v_isSharedCheck_262_;
goto v_resetjp_256_;
}
v_resetjp_256_:
{
lean_object* v___x_260_; 
if (v_isShared_258_ == 0)
{
v___x_260_ = v___x_257_;
goto v_reusejp_259_;
}
else
{
lean_object* v_reuseFailAlloc_261_; 
v_reuseFailAlloc_261_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_261_, 0, v_a_255_);
v___x_260_ = v_reuseFailAlloc_261_;
goto v_reusejp_259_;
}
v_reusejp_259_:
{
return v___x_260_;
}
}
}
}
}
else
{
lean_object* v___x_263_; 
v___x_263_ = lp_aesop_Aesop_ElabM_Context_forAdditionalGlobalRules(v_a_198_, v_a_199_, v___x_223_, v_a_201_);
if (lean_obj_tag(v___x_263_) == 0)
{
lean_object* v_a_264_; lean_object* v___x_265_; lean_object* v___x_266_; lean_object* v___x_267_; 
v_a_264_ = lean_ctor_get(v___x_263_, 0);
lean_inc(v_a_264_);
lean_dec_ref_known(v___x_263_, 1);
v___x_265_ = lean_unsigned_to_nat(0u);
v___x_266_ = l_Lean_Syntax_getArg(v___x_225_, v___x_265_);
lean_dec(v___x_225_);
v___x_267_ = lp_aesop_Aesop_Frontend_RuleExpr_elab(v___x_266_, v_a_264_, v_a_196_, v_a_197_, v_a_198_, v_a_199_, v___x_223_, v_a_201_);
lean_dec_ref_known(v___x_223_, 14);
lean_dec(v_a_264_);
if (lean_obj_tag(v___x_267_) == 0)
{
lean_object* v_a_268_; lean_object* v___x_270_; uint8_t v_isShared_271_; uint8_t v_isSharedCheck_277_; 
v_a_268_ = lean_ctor_get(v___x_267_, 0);
v_isSharedCheck_277_ = !lean_is_exclusive(v___x_267_);
if (v_isSharedCheck_277_ == 0)
{
v___x_270_ = v___x_267_;
v_isShared_271_ = v_isSharedCheck_277_;
goto v_resetjp_269_;
}
else
{
lean_inc(v_a_268_);
lean_dec(v___x_267_);
v___x_270_ = lean_box(0);
v_isShared_271_ = v_isSharedCheck_277_;
goto v_resetjp_269_;
}
v_resetjp_269_:
{
lean_object* v___x_272_; lean_object* v___x_273_; lean_object* v___x_275_; 
v___x_272_ = lean_mk_empty_array_with_capacity(v___x_224_);
v___x_273_ = lean_array_push(v___x_272_, v_a_268_);
if (v_isShared_271_ == 0)
{
lean_ctor_set(v___x_270_, 0, v___x_273_);
v___x_275_ = v___x_270_;
goto v_reusejp_274_;
}
else
{
lean_object* v_reuseFailAlloc_276_; 
v_reuseFailAlloc_276_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_276_, 0, v___x_273_);
v___x_275_ = v_reuseFailAlloc_276_;
goto v_reusejp_274_;
}
v_reusejp_274_:
{
return v___x_275_;
}
}
}
else
{
lean_object* v_a_278_; lean_object* v___x_280_; uint8_t v_isShared_281_; uint8_t v_isSharedCheck_285_; 
v_a_278_ = lean_ctor_get(v___x_267_, 0);
v_isSharedCheck_285_ = !lean_is_exclusive(v___x_267_);
if (v_isSharedCheck_285_ == 0)
{
v___x_280_ = v___x_267_;
v_isShared_281_ = v_isSharedCheck_285_;
goto v_resetjp_279_;
}
else
{
lean_inc(v_a_278_);
lean_dec(v___x_267_);
v___x_280_ = lean_box(0);
v_isShared_281_ = v_isSharedCheck_285_;
goto v_resetjp_279_;
}
v_resetjp_279_:
{
lean_object* v___x_283_; 
if (v_isShared_281_ == 0)
{
v___x_283_ = v___x_280_;
goto v_reusejp_282_;
}
else
{
lean_object* v_reuseFailAlloc_284_; 
v_reuseFailAlloc_284_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_284_, 0, v_a_278_);
v___x_283_ = v_reuseFailAlloc_284_;
goto v_reusejp_282_;
}
v_reusejp_282_:
{
return v___x_283_;
}
}
}
}
else
{
lean_object* v_a_286_; lean_object* v___x_288_; uint8_t v_isShared_289_; uint8_t v_isSharedCheck_293_; 
lean_dec(v___x_225_);
lean_dec_ref_known(v___x_223_, 14);
v_a_286_ = lean_ctor_get(v___x_263_, 0);
v_isSharedCheck_293_ = !lean_is_exclusive(v___x_263_);
if (v_isSharedCheck_293_ == 0)
{
v___x_288_ = v___x_263_;
v_isShared_289_ = v_isSharedCheck_293_;
goto v_resetjp_287_;
}
else
{
lean_inc(v_a_286_);
lean_dec(v___x_263_);
v___x_288_ = lean_box(0);
v_isShared_289_ = v_isSharedCheck_293_;
goto v_resetjp_287_;
}
v_resetjp_287_:
{
lean_object* v___x_291_; 
if (v_isShared_289_ == 0)
{
v___x_291_ = v___x_288_;
goto v_reusejp_290_;
}
else
{
lean_object* v_reuseFailAlloc_292_; 
v_reuseFailAlloc_292_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_292_, 0, v_a_286_);
v___x_291_ = v_reuseFailAlloc_292_;
goto v_reusejp_290_;
}
v_reusejp_290_:
{
return v___x_291_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_AttrConfig_elab___boxed(lean_object* v_stx_294_, lean_object* v_a_295_, lean_object* v_a_296_, lean_object* v_a_297_, lean_object* v_a_298_, lean_object* v_a_299_, lean_object* v_a_300_, lean_object* v_a_301_){
_start:
{
lean_object* v_res_302_; 
v_res_302_ = lp_aesop_Aesop_Frontend_AttrConfig_elab(v_stx_294_, v_a_295_, v_a_296_, v_a_297_, v_a_298_, v_a_299_, v_a_300_);
lean_dec(v_a_300_);
lean_dec_ref(v_a_299_);
lean_dec(v_a_298_);
lean_dec_ref(v_a_297_);
lean_dec(v_a_296_);
lean_dec_ref(v_a_295_);
return v_res_302_;
}
}
LEAN_EXPORT uint8_t lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___lam__0_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2_(lean_object* v_x_303_){
_start:
{
uint8_t v___x_304_; 
v___x_304_ = 0;
return v___x_304_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___lam__0_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2____boxed(lean_object* v_x_305_){
_start:
{
uint8_t v_res_306_; lean_object* v_r_307_; 
v_res_306_ = lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___lam__0_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2_(v_x_305_);
lean_dec(v_x_305_);
v_r_307_ = lean_box(v_res_306_);
return v_r_307_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_modifyGetGlobalRuleSet___at___00__private_Aesop_Frontend_Extension_0__Aesop_Frontend_eraseGlobalRules_go___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0_spec__1_spec__3___redArg___lam__2(lean_object* v_x_308_){
_start:
{
lean_object* v___x_309_; 
v___x_309_ = lp_aesop_Aesop_instInhabitedBaseRuleSet_default;
return v___x_309_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_modifyGetGlobalRuleSet___at___00__private_Aesop_Frontend_Extension_0__Aesop_Frontend_eraseGlobalRules_go___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0_spec__1_spec__3___redArg___lam__2___boxed(lean_object* v_x_310_){
_start:
{
lean_object* v_res_311_; 
v_res_311_ = lp_aesop_Aesop_Frontend_modifyGetGlobalRuleSet___at___00__private_Aesop_Frontend_Extension_0__Aesop_Frontend_eraseGlobalRules_go___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0_spec__1_spec__3___redArg___lam__2(v_x_310_);
lean_dec_ref(v_x_310_);
return v_res_311_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_modifyGetGlobalRuleSet___at___00__private_Aesop_Frontend_Extension_0__Aesop_Frontend_eraseGlobalRules_go___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0_spec__1_spec__3___redArg___lam__0(lean_object* v_x_312_){
_start:
{
lean_object* v___x_313_; 
v___x_313_ = l_Lean_Meta_Simp_instInhabitedSimprocs_default;
return v___x_313_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_modifyGetGlobalRuleSet___at___00__private_Aesop_Frontend_Extension_0__Aesop_Frontend_eraseGlobalRules_go___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0_spec__1_spec__3___redArg___lam__0___boxed(lean_object* v_x_314_){
_start:
{
lean_object* v_res_315_; 
v_res_315_ = lp_aesop_Aesop_Frontend_modifyGetGlobalRuleSet___at___00__private_Aesop_Frontend_Extension_0__Aesop_Frontend_eraseGlobalRules_go___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0_spec__1_spec__3___redArg___lam__0(v_x_314_);
lean_dec_ref(v_x_314_);
return v_res_315_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__9_spec__12_spec__18___redArg(lean_object* v_a_316_, lean_object* v_x_317_){
_start:
{
if (lean_obj_tag(v_x_317_) == 0)
{
lean_object* v___x_318_; 
v___x_318_ = lean_box(0);
return v___x_318_;
}
else
{
lean_object* v_key_319_; lean_object* v_value_320_; lean_object* v_tail_321_; uint8_t v___x_322_; 
v_key_319_ = lean_ctor_get(v_x_317_, 0);
v_value_320_ = lean_ctor_get(v_x_317_, 1);
v_tail_321_ = lean_ctor_get(v_x_317_, 2);
v___x_322_ = lean_name_eq(v_key_319_, v_a_316_);
if (v___x_322_ == 0)
{
v_x_317_ = v_tail_321_;
goto _start;
}
else
{
lean_object* v___x_324_; 
lean_inc(v_value_320_);
v___x_324_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_324_, 0, v_value_320_);
return v___x_324_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__9_spec__12_spec__18___redArg___boxed(lean_object* v_a_325_, lean_object* v_x_326_){
_start:
{
lean_object* v_res_327_; 
v_res_327_ = lp_aesop_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__9_spec__12_spec__18___redArg(v_a_325_, v_x_326_);
lean_dec(v_x_326_);
lean_dec(v_a_325_);
return v_res_327_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__9_spec__12___redArg(lean_object* v_m_328_, lean_object* v_a_329_){
_start:
{
lean_object* v_buckets_330_; lean_object* v___x_331_; uint64_t v___y_333_; 
v_buckets_330_ = lean_ctor_get(v_m_328_, 1);
v___x_331_ = lean_array_get_size(v_buckets_330_);
if (lean_obj_tag(v_a_329_) == 0)
{
uint64_t v___x_347_; 
v___x_347_ = 1723ULL;
v___y_333_ = v___x_347_;
goto v___jp_332_;
}
else
{
uint64_t v_hash_348_; 
v_hash_348_ = lean_ctor_get_uint64(v_a_329_, sizeof(void*)*2);
v___y_333_ = v_hash_348_;
goto v___jp_332_;
}
v___jp_332_:
{
uint64_t v___x_334_; uint64_t v___x_335_; uint64_t v_fold_336_; uint64_t v___x_337_; uint64_t v___x_338_; uint64_t v___x_339_; size_t v___x_340_; size_t v___x_341_; size_t v___x_342_; size_t v___x_343_; size_t v___x_344_; lean_object* v___x_345_; lean_object* v___x_346_; 
v___x_334_ = 32ULL;
v___x_335_ = lean_uint64_shift_right(v___y_333_, v___x_334_);
v_fold_336_ = lean_uint64_xor(v___y_333_, v___x_335_);
v___x_337_ = 16ULL;
v___x_338_ = lean_uint64_shift_right(v_fold_336_, v___x_337_);
v___x_339_ = lean_uint64_xor(v_fold_336_, v___x_338_);
v___x_340_ = lean_uint64_to_usize(v___x_339_);
v___x_341_ = lean_usize_of_nat(v___x_331_);
v___x_342_ = ((size_t)1ULL);
v___x_343_ = lean_usize_sub(v___x_341_, v___x_342_);
v___x_344_ = lean_usize_land(v___x_340_, v___x_343_);
v___x_345_ = lean_array_uget_borrowed(v_buckets_330_, v___x_344_);
v___x_346_ = lp_aesop_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__9_spec__12_spec__18___redArg(v_a_329_, v___x_345_);
return v___x_346_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__9_spec__12___redArg___boxed(lean_object* v_m_349_, lean_object* v_a_350_){
_start:
{
lean_object* v_res_351_; 
v_res_351_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__9_spec__12___redArg(v_m_349_, v_a_350_);
lean_dec(v_a_350_);
lean_dec_ref(v_m_349_);
return v_res_351_;
}
}
static lean_object* _init_lp_aesop_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0_spec__0_spec__1___closed__0(void){
_start:
{
lean_object* v___x_352_; 
v___x_352_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_352_;
}
}
static lean_object* _init_lp_aesop_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0_spec__0_spec__1___closed__1(void){
_start:
{
lean_object* v___x_353_; lean_object* v___x_354_; 
v___x_353_ = lean_obj_once(&lp_aesop_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0_spec__0_spec__1___closed__0, &lp_aesop_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0_spec__0_spec__1___closed__0_once, _init_lp_aesop_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0_spec__0_spec__1___closed__0);
v___x_354_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_354_, 0, v___x_353_);
return v___x_354_;
}
}
static lean_object* _init_lp_aesop_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0_spec__0_spec__1___closed__2(void){
_start:
{
lean_object* v___x_355_; lean_object* v___x_356_; lean_object* v___x_357_; 
v___x_355_ = lean_obj_once(&lp_aesop_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0_spec__0_spec__1___closed__1, &lp_aesop_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0_spec__0_spec__1___closed__1_once, _init_lp_aesop_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0_spec__0_spec__1___closed__1);
v___x_356_ = lean_unsigned_to_nat(0u);
v___x_357_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v___x_357_, 0, v___x_356_);
lean_ctor_set(v___x_357_, 1, v___x_356_);
lean_ctor_set(v___x_357_, 2, v___x_356_);
lean_ctor_set(v___x_357_, 3, v___x_356_);
lean_ctor_set(v___x_357_, 4, v___x_355_);
lean_ctor_set(v___x_357_, 5, v___x_355_);
lean_ctor_set(v___x_357_, 6, v___x_355_);
lean_ctor_set(v___x_357_, 7, v___x_355_);
lean_ctor_set(v___x_357_, 8, v___x_355_);
lean_ctor_set(v___x_357_, 9, v___x_355_);
return v___x_357_;
}
}
static lean_object* _init_lp_aesop_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0_spec__0_spec__1___closed__3(void){
_start:
{
lean_object* v___x_358_; lean_object* v___x_359_; lean_object* v___x_360_; 
v___x_358_ = lean_unsigned_to_nat(32u);
v___x_359_ = lean_mk_empty_array_with_capacity(v___x_358_);
v___x_360_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_360_, 0, v___x_359_);
return v___x_360_;
}
}
static lean_object* _init_lp_aesop_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0_spec__0_spec__1___closed__4(void){
_start:
{
size_t v___x_361_; lean_object* v___x_362_; lean_object* v___x_363_; lean_object* v___x_364_; lean_object* v___x_365_; lean_object* v___x_366_; 
v___x_361_ = ((size_t)5ULL);
v___x_362_ = lean_unsigned_to_nat(0u);
v___x_363_ = lean_unsigned_to_nat(32u);
v___x_364_ = lean_mk_empty_array_with_capacity(v___x_363_);
v___x_365_ = lean_obj_once(&lp_aesop_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0_spec__0_spec__1___closed__3, &lp_aesop_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0_spec__0_spec__1___closed__3_once, _init_lp_aesop_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0_spec__0_spec__1___closed__3);
v___x_366_ = lean_alloc_ctor(0, 4, sizeof(size_t)*1);
lean_ctor_set(v___x_366_, 0, v___x_365_);
lean_ctor_set(v___x_366_, 1, v___x_364_);
lean_ctor_set(v___x_366_, 2, v___x_362_);
lean_ctor_set(v___x_366_, 3, v___x_362_);
lean_ctor_set_usize(v___x_366_, 4, v___x_361_);
return v___x_366_;
}
}
static lean_object* _init_lp_aesop_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0_spec__0_spec__1___closed__5(void){
_start:
{
lean_object* v___x_367_; lean_object* v___x_368_; lean_object* v___x_369_; lean_object* v___x_370_; 
v___x_367_ = lean_box(1);
v___x_368_ = lean_obj_once(&lp_aesop_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0_spec__0_spec__1___closed__4, &lp_aesop_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0_spec__0_spec__1___closed__4_once, _init_lp_aesop_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0_spec__0_spec__1___closed__4);
v___x_369_ = lean_obj_once(&lp_aesop_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0_spec__0_spec__1___closed__1, &lp_aesop_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0_spec__0_spec__1___closed__1_once, _init_lp_aesop_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0_spec__0_spec__1___closed__1);
v___x_370_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_370_, 0, v___x_369_);
lean_ctor_set(v___x_370_, 1, v___x_368_);
lean_ctor_set(v___x_370_, 2, v___x_367_);
return v___x_370_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0_spec__0_spec__1(lean_object* v_msgData_371_, lean_object* v___y_372_, lean_object* v___y_373_){
_start:
{
lean_object* v___x_375_; lean_object* v_env_376_; lean_object* v_options_377_; lean_object* v___x_378_; lean_object* v___x_379_; lean_object* v___x_380_; lean_object* v___x_381_; lean_object* v___x_382_; 
v___x_375_ = lean_st_ref_get(v___y_373_);
v_env_376_ = lean_ctor_get(v___x_375_, 0);
lean_inc_ref(v_env_376_);
lean_dec(v___x_375_);
v_options_377_ = lean_ctor_get(v___y_372_, 2);
v___x_378_ = lean_obj_once(&lp_aesop_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0_spec__0_spec__1___closed__2, &lp_aesop_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0_spec__0_spec__1___closed__2_once, _init_lp_aesop_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0_spec__0_spec__1___closed__2);
v___x_379_ = lean_obj_once(&lp_aesop_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0_spec__0_spec__1___closed__5, &lp_aesop_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0_spec__0_spec__1___closed__5_once, _init_lp_aesop_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0_spec__0_spec__1___closed__5);
lean_inc_ref(v_options_377_);
v___x_380_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_380_, 0, v_env_376_);
lean_ctor_set(v___x_380_, 1, v___x_378_);
lean_ctor_set(v___x_380_, 2, v___x_379_);
lean_ctor_set(v___x_380_, 3, v_options_377_);
v___x_381_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_381_, 0, v___x_380_);
lean_ctor_set(v___x_381_, 1, v_msgData_371_);
v___x_382_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_382_, 0, v___x_381_);
return v___x_382_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0_spec__0_spec__1___boxed(lean_object* v_msgData_383_, lean_object* v___y_384_, lean_object* v___y_385_, lean_object* v___y_386_){
_start:
{
lean_object* v_res_387_; 
v_res_387_ = lp_aesop_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0_spec__0_spec__1(v_msgData_383_, v___y_384_, v___y_385_);
lean_dec(v___y_385_);
lean_dec_ref(v___y_384_);
return v_res_387_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0_spec__0___redArg(lean_object* v_msg_388_, lean_object* v___y_389_, lean_object* v___y_390_){
_start:
{
lean_object* v_ref_392_; lean_object* v___x_393_; lean_object* v_a_394_; lean_object* v___x_396_; uint8_t v_isShared_397_; uint8_t v_isSharedCheck_402_; 
v_ref_392_ = lean_ctor_get(v___y_389_, 5);
v___x_393_ = lp_aesop_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0_spec__0_spec__1(v_msg_388_, v___y_389_, v___y_390_);
v_a_394_ = lean_ctor_get(v___x_393_, 0);
v_isSharedCheck_402_ = !lean_is_exclusive(v___x_393_);
if (v_isSharedCheck_402_ == 0)
{
v___x_396_ = v___x_393_;
v_isShared_397_ = v_isSharedCheck_402_;
goto v_resetjp_395_;
}
else
{
lean_inc(v_a_394_);
lean_dec(v___x_393_);
v___x_396_ = lean_box(0);
v_isShared_397_ = v_isSharedCheck_402_;
goto v_resetjp_395_;
}
v_resetjp_395_:
{
lean_object* v___x_398_; lean_object* v___x_400_; 
lean_inc(v_ref_392_);
v___x_398_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_398_, 0, v_ref_392_);
lean_ctor_set(v___x_398_, 1, v_a_394_);
if (v_isShared_397_ == 0)
{
lean_ctor_set_tag(v___x_396_, 1);
lean_ctor_set(v___x_396_, 0, v___x_398_);
v___x_400_ = v___x_396_;
goto v_reusejp_399_;
}
else
{
lean_object* v_reuseFailAlloc_401_; 
v_reuseFailAlloc_401_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_401_, 0, v___x_398_);
v___x_400_ = v_reuseFailAlloc_401_;
goto v_reusejp_399_;
}
v_reusejp_399_:
{
return v___x_400_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0_spec__0___redArg___boxed(lean_object* v_msg_403_, lean_object* v___y_404_, lean_object* v___y_405_, lean_object* v___y_406_){
_start:
{
lean_object* v_res_407_; 
v_res_407_ = lp_aesop_Lean_throwError___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0_spec__0___redArg(v_msg_403_, v___y_404_, v___y_405_);
lean_dec(v___y_405_);
lean_dec_ref(v___y_404_);
return v_res_407_;
}
}
static lean_object* _init_lp_aesop_Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__9___closed__1(void){
_start:
{
lean_object* v___x_409_; lean_object* v___x_410_; 
v___x_409_ = ((lean_object*)(lp_aesop_Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__9___closed__0));
v___x_410_ = l_Lean_stringToMessageData(v___x_409_);
return v___x_410_;
}
}
static lean_object* _init_lp_aesop_Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__9___closed__3(void){
_start:
{
lean_object* v___x_412_; lean_object* v___x_413_; 
v___x_412_ = ((lean_object*)(lp_aesop_Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__9___closed__2));
v___x_413_ = l_Lean_stringToMessageData(v___x_412_);
return v___x_413_;
}
}
static lean_object* _init_lp_aesop_Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__9___closed__5(void){
_start:
{
lean_object* v___x_415_; lean_object* v___x_416_; 
v___x_415_ = ((lean_object*)(lp_aesop_Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__9___closed__4));
v___x_416_ = l_Lean_stringToMessageData(v___x_415_);
return v___x_416_;
}
}
static lean_object* _init_lp_aesop_Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__9___closed__7(void){
_start:
{
lean_object* v___x_418_; lean_object* v___x_419_; 
v___x_418_ = ((lean_object*)(lp_aesop_Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__9___closed__6));
v___x_419_ = l_Lean_stringToMessageData(v___x_418_);
return v___x_419_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__9(lean_object* v_rsName_420_, lean_object* v___y_421_, lean_object* v___y_422_){
_start:
{
lean_object* v___x_424_; 
v___x_424_ = lp_aesop_Aesop_getDeclaredRuleSets();
if (lean_obj_tag(v___x_424_) == 0)
{
lean_object* v_a_425_; lean_object* v___x_426_; 
v_a_425_ = lean_ctor_get(v___x_424_, 0);
lean_inc(v_a_425_);
lean_dec_ref_known(v___x_424_, 1);
v___x_426_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__9_spec__12___redArg(v_a_425_, v_rsName_420_);
lean_dec(v_a_425_);
if (lean_obj_tag(v___x_426_) == 1)
{
lean_object* v_val_427_; lean_object* v_snd_428_; lean_object* v_fst_429_; lean_object* v___x_431_; uint8_t v_isShared_432_; uint8_t v_isSharedCheck_499_; 
lean_dec(v_rsName_420_);
v_val_427_ = lean_ctor_get(v___x_426_, 0);
lean_inc(v_val_427_);
lean_dec_ref_known(v___x_426_, 1);
v_snd_428_ = lean_ctor_get(v_val_427_, 1);
v_fst_429_ = lean_ctor_get(v_val_427_, 0);
v_isSharedCheck_499_ = !lean_is_exclusive(v_val_427_);
if (v_isSharedCheck_499_ == 0)
{
v___x_431_ = v_val_427_;
v_isShared_432_ = v_isSharedCheck_499_;
goto v_resetjp_430_;
}
else
{
lean_inc(v_snd_428_);
lean_inc(v_fst_429_);
lean_dec(v_val_427_);
v___x_431_ = lean_box(0);
v_isShared_432_ = v_isSharedCheck_499_;
goto v_resetjp_430_;
}
v_resetjp_430_:
{
lean_object* v_fst_433_; lean_object* v_snd_434_; lean_object* v___x_436_; uint8_t v_isShared_437_; uint8_t v_isSharedCheck_498_; 
v_fst_433_ = lean_ctor_get(v_snd_428_, 0);
v_snd_434_ = lean_ctor_get(v_snd_428_, 1);
v_isSharedCheck_498_ = !lean_is_exclusive(v_snd_428_);
if (v_isSharedCheck_498_ == 0)
{
v___x_436_ = v_snd_428_;
v_isShared_437_ = v_isSharedCheck_498_;
goto v_resetjp_435_;
}
else
{
lean_inc(v_snd_434_);
lean_inc(v_fst_433_);
lean_dec(v_snd_428_);
v___x_436_ = lean_box(0);
v_isShared_437_ = v_isSharedCheck_498_;
goto v_resetjp_435_;
}
v_resetjp_435_:
{
lean_object* v___x_438_; 
v___x_438_ = l_Lean_Meta_getSimpExtension_x3f(v_fst_433_, v___y_421_, v___y_422_);
if (lean_obj_tag(v___x_438_) == 0)
{
lean_object* v_a_439_; 
v_a_439_ = lean_ctor_get(v___x_438_, 0);
lean_inc(v_a_439_);
lean_dec_ref_known(v___x_438_, 1);
if (lean_obj_tag(v_a_439_) == 1)
{
lean_object* v_val_440_; lean_object* v___x_442_; uint8_t v_isShared_443_; uint8_t v_isSharedCheck_483_; 
v_val_440_ = lean_ctor_get(v_a_439_, 0);
v_isSharedCheck_483_ = !lean_is_exclusive(v_a_439_);
if (v_isSharedCheck_483_ == 0)
{
v___x_442_ = v_a_439_;
v_isShared_443_ = v_isSharedCheck_483_;
goto v_resetjp_441_;
}
else
{
lean_inc(v_val_440_);
lean_dec(v_a_439_);
v___x_442_ = lean_box(0);
v_isShared_443_ = v_isSharedCheck_483_;
goto v_resetjp_441_;
}
v_resetjp_441_:
{
lean_object* v___x_444_; 
lean_inc(v_fst_433_);
v___x_444_ = l_Lean_Meta_Simp_getSimprocExtension_x3f(v_fst_433_);
if (lean_obj_tag(v___x_444_) == 0)
{
lean_object* v_a_445_; lean_object* v___x_447_; uint8_t v_isShared_448_; uint8_t v_isSharedCheck_467_; 
lean_del_object(v___x_442_);
v_a_445_ = lean_ctor_get(v___x_444_, 0);
v_isSharedCheck_467_ = !lean_is_exclusive(v___x_444_);
if (v_isSharedCheck_467_ == 0)
{
v___x_447_ = v___x_444_;
v_isShared_448_ = v_isSharedCheck_467_;
goto v_resetjp_446_;
}
else
{
lean_inc(v_a_445_);
lean_dec(v___x_444_);
v___x_447_ = lean_box(0);
v_isShared_448_ = v_isSharedCheck_467_;
goto v_resetjp_446_;
}
v_resetjp_446_:
{
if (lean_obj_tag(v_a_445_) == 1)
{
lean_object* v_val_449_; lean_object* v___x_451_; 
v_val_449_ = lean_ctor_get(v_a_445_, 0);
lean_inc(v_val_449_);
lean_dec_ref_known(v_a_445_, 1);
if (v_isShared_437_ == 0)
{
lean_ctor_set(v___x_436_, 1, v_val_449_);
lean_ctor_set(v___x_436_, 0, v_snd_434_);
v___x_451_ = v___x_436_;
goto v_reusejp_450_;
}
else
{
lean_object* v_reuseFailAlloc_460_; 
v_reuseFailAlloc_460_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_460_, 0, v_snd_434_);
lean_ctor_set(v_reuseFailAlloc_460_, 1, v_val_449_);
v___x_451_ = v_reuseFailAlloc_460_;
goto v_reusejp_450_;
}
v_reusejp_450_:
{
lean_object* v___x_453_; 
if (v_isShared_432_ == 0)
{
lean_ctor_set(v___x_431_, 1, v___x_451_);
lean_ctor_set(v___x_431_, 0, v_val_440_);
v___x_453_ = v___x_431_;
goto v_reusejp_452_;
}
else
{
lean_object* v_reuseFailAlloc_459_; 
v_reuseFailAlloc_459_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_459_, 0, v_val_440_);
lean_ctor_set(v_reuseFailAlloc_459_, 1, v___x_451_);
v___x_453_ = v_reuseFailAlloc_459_;
goto v_reusejp_452_;
}
v_reusejp_452_:
{
lean_object* v___x_454_; lean_object* v___x_455_; lean_object* v___x_457_; 
v___x_454_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_454_, 0, v_fst_433_);
lean_ctor_set(v___x_454_, 1, v___x_453_);
v___x_455_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_455_, 0, v_fst_429_);
lean_ctor_set(v___x_455_, 1, v___x_454_);
if (v_isShared_448_ == 0)
{
lean_ctor_set(v___x_447_, 0, v___x_455_);
v___x_457_ = v___x_447_;
goto v_reusejp_456_;
}
else
{
lean_object* v_reuseFailAlloc_458_; 
v_reuseFailAlloc_458_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_458_, 0, v___x_455_);
v___x_457_ = v_reuseFailAlloc_458_;
goto v_reusejp_456_;
}
v_reusejp_456_:
{
return v___x_457_;
}
}
}
}
else
{
lean_object* v___x_461_; lean_object* v___x_462_; lean_object* v___x_463_; lean_object* v___x_464_; lean_object* v___x_465_; lean_object* v___x_466_; 
lean_del_object(v___x_447_);
lean_dec(v_a_445_);
lean_dec(v_val_440_);
lean_del_object(v___x_436_);
lean_dec(v_snd_434_);
lean_del_object(v___x_431_);
lean_dec(v_fst_429_);
v___x_461_ = lean_obj_once(&lp_aesop_Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__9___closed__1, &lp_aesop_Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__9___closed__1_once, _init_lp_aesop_Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__9___closed__1);
v___x_462_ = l_Lean_MessageData_ofName(v_fst_433_);
v___x_463_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_463_, 0, v___x_461_);
lean_ctor_set(v___x_463_, 1, v___x_462_);
v___x_464_ = lean_obj_once(&lp_aesop_Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__9___closed__3, &lp_aesop_Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__9___closed__3_once, _init_lp_aesop_Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__9___closed__3);
v___x_465_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_465_, 0, v___x_463_);
lean_ctor_set(v___x_465_, 1, v___x_464_);
v___x_466_ = lp_aesop_Lean_throwError___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0_spec__0___redArg(v___x_465_, v___y_421_, v___y_422_);
return v___x_466_;
}
}
}
else
{
lean_object* v_a_468_; lean_object* v___x_470_; uint8_t v_isShared_471_; uint8_t v_isSharedCheck_482_; 
lean_dec(v_val_440_);
lean_del_object(v___x_436_);
lean_dec(v_snd_434_);
lean_dec(v_fst_433_);
lean_del_object(v___x_431_);
lean_dec(v_fst_429_);
v_a_468_ = lean_ctor_get(v___x_444_, 0);
v_isSharedCheck_482_ = !lean_is_exclusive(v___x_444_);
if (v_isSharedCheck_482_ == 0)
{
v___x_470_ = v___x_444_;
v_isShared_471_ = v_isSharedCheck_482_;
goto v_resetjp_469_;
}
else
{
lean_inc(v_a_468_);
lean_dec(v___x_444_);
v___x_470_ = lean_box(0);
v_isShared_471_ = v_isSharedCheck_482_;
goto v_resetjp_469_;
}
v_resetjp_469_:
{
lean_object* v_ref_472_; lean_object* v___x_473_; lean_object* v___x_475_; 
v_ref_472_ = lean_ctor_get(v___y_421_, 5);
v___x_473_ = lean_io_error_to_string(v_a_468_);
if (v_isShared_443_ == 0)
{
lean_ctor_set_tag(v___x_442_, 3);
lean_ctor_set(v___x_442_, 0, v___x_473_);
v___x_475_ = v___x_442_;
goto v_reusejp_474_;
}
else
{
lean_object* v_reuseFailAlloc_481_; 
v_reuseFailAlloc_481_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v_reuseFailAlloc_481_, 0, v___x_473_);
v___x_475_ = v_reuseFailAlloc_481_;
goto v_reusejp_474_;
}
v_reusejp_474_:
{
lean_object* v___x_476_; lean_object* v___x_477_; lean_object* v___x_479_; 
v___x_476_ = l_Lean_MessageData_ofFormat(v___x_475_);
lean_inc(v_ref_472_);
v___x_477_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_477_, 0, v_ref_472_);
lean_ctor_set(v___x_477_, 1, v___x_476_);
if (v_isShared_471_ == 0)
{
lean_ctor_set(v___x_470_, 0, v___x_477_);
v___x_479_ = v___x_470_;
goto v_reusejp_478_;
}
else
{
lean_object* v_reuseFailAlloc_480_; 
v_reuseFailAlloc_480_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_480_, 0, v___x_477_);
v___x_479_ = v_reuseFailAlloc_480_;
goto v_reusejp_478_;
}
v_reusejp_478_:
{
return v___x_479_;
}
}
}
}
}
}
else
{
lean_object* v___x_484_; lean_object* v___x_485_; lean_object* v___x_486_; lean_object* v___x_487_; lean_object* v___x_488_; lean_object* v___x_489_; 
lean_dec(v_a_439_);
lean_del_object(v___x_436_);
lean_dec(v_snd_434_);
lean_del_object(v___x_431_);
lean_dec(v_fst_429_);
v___x_484_ = lean_obj_once(&lp_aesop_Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__9___closed__1, &lp_aesop_Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__9___closed__1_once, _init_lp_aesop_Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__9___closed__1);
v___x_485_ = l_Lean_MessageData_ofName(v_fst_433_);
v___x_486_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_486_, 0, v___x_484_);
lean_ctor_set(v___x_486_, 1, v___x_485_);
v___x_487_ = lean_obj_once(&lp_aesop_Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__9___closed__3, &lp_aesop_Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__9___closed__3_once, _init_lp_aesop_Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__9___closed__3);
v___x_488_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_488_, 0, v___x_486_);
lean_ctor_set(v___x_488_, 1, v___x_487_);
v___x_489_ = lp_aesop_Lean_throwError___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0_spec__0___redArg(v___x_488_, v___y_421_, v___y_422_);
return v___x_489_;
}
}
else
{
lean_object* v_a_490_; lean_object* v___x_492_; uint8_t v_isShared_493_; uint8_t v_isSharedCheck_497_; 
lean_del_object(v___x_436_);
lean_dec(v_snd_434_);
lean_dec(v_fst_433_);
lean_del_object(v___x_431_);
lean_dec(v_fst_429_);
v_a_490_ = lean_ctor_get(v___x_438_, 0);
v_isSharedCheck_497_ = !lean_is_exclusive(v___x_438_);
if (v_isSharedCheck_497_ == 0)
{
v___x_492_ = v___x_438_;
v_isShared_493_ = v_isSharedCheck_497_;
goto v_resetjp_491_;
}
else
{
lean_inc(v_a_490_);
lean_dec(v___x_438_);
v___x_492_ = lean_box(0);
v_isShared_493_ = v_isSharedCheck_497_;
goto v_resetjp_491_;
}
v_resetjp_491_:
{
lean_object* v___x_495_; 
if (v_isShared_493_ == 0)
{
v___x_495_ = v___x_492_;
goto v_reusejp_494_;
}
else
{
lean_object* v_reuseFailAlloc_496_; 
v_reuseFailAlloc_496_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_496_, 0, v_a_490_);
v___x_495_ = v_reuseFailAlloc_496_;
goto v_reusejp_494_;
}
v_reusejp_494_:
{
return v___x_495_;
}
}
}
}
}
}
else
{
lean_object* v___x_500_; lean_object* v___x_501_; lean_object* v___x_502_; lean_object* v___x_503_; lean_object* v___x_504_; lean_object* v___x_505_; 
lean_dec(v___x_426_);
v___x_500_ = lean_obj_once(&lp_aesop_Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__9___closed__5, &lp_aesop_Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__9___closed__5_once, _init_lp_aesop_Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__9___closed__5);
v___x_501_ = l_Lean_MessageData_ofName(v_rsName_420_);
v___x_502_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_502_, 0, v___x_500_);
lean_ctor_set(v___x_502_, 1, v___x_501_);
v___x_503_ = lean_obj_once(&lp_aesop_Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__9___closed__7, &lp_aesop_Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__9___closed__7_once, _init_lp_aesop_Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__9___closed__7);
v___x_504_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_504_, 0, v___x_502_);
lean_ctor_set(v___x_504_, 1, v___x_503_);
v___x_505_ = lp_aesop_Lean_throwError___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0_spec__0___redArg(v___x_504_, v___y_421_, v___y_422_);
return v___x_505_;
}
}
else
{
lean_object* v_a_506_; lean_object* v___x_508_; uint8_t v_isShared_509_; uint8_t v_isSharedCheck_518_; 
lean_dec(v_rsName_420_);
v_a_506_ = lean_ctor_get(v___x_424_, 0);
v_isSharedCheck_518_ = !lean_is_exclusive(v___x_424_);
if (v_isSharedCheck_518_ == 0)
{
v___x_508_ = v___x_424_;
v_isShared_509_ = v_isSharedCheck_518_;
goto v_resetjp_507_;
}
else
{
lean_inc(v_a_506_);
lean_dec(v___x_424_);
v___x_508_ = lean_box(0);
v_isShared_509_ = v_isSharedCheck_518_;
goto v_resetjp_507_;
}
v_resetjp_507_:
{
lean_object* v_ref_510_; lean_object* v___x_511_; lean_object* v___x_512_; lean_object* v___x_513_; lean_object* v___x_514_; lean_object* v___x_516_; 
v_ref_510_ = lean_ctor_get(v___y_421_, 5);
v___x_511_ = lean_io_error_to_string(v_a_506_);
v___x_512_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_512_, 0, v___x_511_);
v___x_513_ = l_Lean_MessageData_ofFormat(v___x_512_);
lean_inc(v_ref_510_);
v___x_514_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_514_, 0, v_ref_510_);
lean_ctor_set(v___x_514_, 1, v___x_513_);
if (v_isShared_509_ == 0)
{
lean_ctor_set(v___x_508_, 0, v___x_514_);
v___x_516_ = v___x_508_;
goto v_reusejp_515_;
}
else
{
lean_object* v_reuseFailAlloc_517_; 
v_reuseFailAlloc_517_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_517_, 0, v___x_514_);
v___x_516_ = v_reuseFailAlloc_517_;
goto v_reusejp_515_;
}
v_reusejp_515_:
{
return v___x_516_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__9___boxed(lean_object* v_rsName_519_, lean_object* v___y_520_, lean_object* v___y_521_, lean_object* v___y_522_){
_start:
{
lean_object* v_res_523_; 
v_res_523_ = lp_aesop_Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__9(v_rsName_519_, v___y_520_, v___y_521_);
lean_dec(v___y_521_);
lean_dec_ref(v___y_520_);
return v_res_523_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_modifyGetGlobalRuleSet___at___00__private_Aesop_Frontend_Extension_0__Aesop_Frontend_eraseGlobalRules_go___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0_spec__1_spec__3___redArg___lam__1(lean_object* v_x_524_){
_start:
{
lean_object* v___x_525_; 
v___x_525_ = l_Lean_Meta_instInhabitedSimpTheorems_default;
return v___x_525_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_modifyGetGlobalRuleSet___at___00__private_Aesop_Frontend_Extension_0__Aesop_Frontend_eraseGlobalRules_go___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0_spec__1_spec__3___redArg___lam__1___boxed(lean_object* v_x_526_){
_start:
{
lean_object* v_res_527_; 
v_res_527_ = lp_aesop_Aesop_Frontend_modifyGetGlobalRuleSet___at___00__private_Aesop_Frontend_Extension_0__Aesop_Frontend_eraseGlobalRules_go___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0_spec__1_spec__3___redArg___lam__1(v_x_526_);
lean_dec_ref(v_x_526_);
return v_res_527_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_modifyGetGlobalRuleSet___at___00__private_Aesop_Frontend_Extension_0__Aesop_Frontend_eraseGlobalRules_go___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0_spec__1_spec__3___redArg___lam__3(lean_object* v_snd_528_, lean_object* v_x_529_){
_start:
{
lean_object* v_simpTheorems_530_; 
v_simpTheorems_530_ = lean_ctor_get(v_snd_528_, 1);
lean_inc_ref(v_simpTheorems_530_);
return v_simpTheorems_530_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_modifyGetGlobalRuleSet___at___00__private_Aesop_Frontend_Extension_0__Aesop_Frontend_eraseGlobalRules_go___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0_spec__1_spec__3___redArg___lam__3___boxed(lean_object* v_snd_531_, lean_object* v_x_532_){
_start:
{
lean_object* v_res_533_; 
v_res_533_ = lp_aesop_Aesop_Frontend_modifyGetGlobalRuleSet___at___00__private_Aesop_Frontend_Extension_0__Aesop_Frontend_eraseGlobalRules_go___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0_spec__1_spec__3___redArg___lam__3(v_snd_531_, v_x_532_);
lean_dec_ref(v_x_532_);
lean_dec_ref(v_snd_531_);
return v_res_533_;
}
}
static lean_object* _init_lp_aesop_Lean_setEnv___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__12___redArg___closed__0(void){
_start:
{
lean_object* v___x_534_; 
v___x_534_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_534_;
}
}
static lean_object* _init_lp_aesop_Lean_setEnv___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__12___redArg___closed__1(void){
_start:
{
lean_object* v___x_535_; lean_object* v___x_536_; 
v___x_535_ = lean_obj_once(&lp_aesop_Lean_setEnv___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__12___redArg___closed__0, &lp_aesop_Lean_setEnv___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__12___redArg___closed__0_once, _init_lp_aesop_Lean_setEnv___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__12___redArg___closed__0);
v___x_536_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_536_, 0, v___x_535_);
return v___x_536_;
}
}
static lean_object* _init_lp_aesop_Lean_setEnv___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__12___redArg___closed__2(void){
_start:
{
lean_object* v___x_537_; lean_object* v___x_538_; 
v___x_537_ = lean_obj_once(&lp_aesop_Lean_setEnv___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__12___redArg___closed__1, &lp_aesop_Lean_setEnv___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__12___redArg___closed__1_once, _init_lp_aesop_Lean_setEnv___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__12___redArg___closed__1);
v___x_538_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_538_, 0, v___x_537_);
lean_ctor_set(v___x_538_, 1, v___x_537_);
return v___x_538_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_setEnv___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__12___redArg(lean_object* v_env_539_, lean_object* v___y_540_){
_start:
{
lean_object* v___x_542_; lean_object* v_nextMacroScope_543_; lean_object* v_ngen_544_; lean_object* v_auxDeclNGen_545_; lean_object* v_traceState_546_; lean_object* v_messages_547_; lean_object* v_infoState_548_; lean_object* v_snapshotTasks_549_; lean_object* v___x_551_; uint8_t v_isShared_552_; uint8_t v_isSharedCheck_560_; 
v___x_542_ = lean_st_ref_take(v___y_540_);
v_nextMacroScope_543_ = lean_ctor_get(v___x_542_, 1);
v_ngen_544_ = lean_ctor_get(v___x_542_, 2);
v_auxDeclNGen_545_ = lean_ctor_get(v___x_542_, 3);
v_traceState_546_ = lean_ctor_get(v___x_542_, 4);
v_messages_547_ = lean_ctor_get(v___x_542_, 6);
v_infoState_548_ = lean_ctor_get(v___x_542_, 7);
v_snapshotTasks_549_ = lean_ctor_get(v___x_542_, 8);
v_isSharedCheck_560_ = !lean_is_exclusive(v___x_542_);
if (v_isSharedCheck_560_ == 0)
{
lean_object* v_unused_561_; lean_object* v_unused_562_; 
v_unused_561_ = lean_ctor_get(v___x_542_, 5);
lean_dec(v_unused_561_);
v_unused_562_ = lean_ctor_get(v___x_542_, 0);
lean_dec(v_unused_562_);
v___x_551_ = v___x_542_;
v_isShared_552_ = v_isSharedCheck_560_;
goto v_resetjp_550_;
}
else
{
lean_inc(v_snapshotTasks_549_);
lean_inc(v_infoState_548_);
lean_inc(v_messages_547_);
lean_inc(v_traceState_546_);
lean_inc(v_auxDeclNGen_545_);
lean_inc(v_ngen_544_);
lean_inc(v_nextMacroScope_543_);
lean_dec(v___x_542_);
v___x_551_ = lean_box(0);
v_isShared_552_ = v_isSharedCheck_560_;
goto v_resetjp_550_;
}
v_resetjp_550_:
{
lean_object* v___x_553_; lean_object* v___x_555_; 
v___x_553_ = lean_obj_once(&lp_aesop_Lean_setEnv___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__12___redArg___closed__2, &lp_aesop_Lean_setEnv___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__12___redArg___closed__2_once, _init_lp_aesop_Lean_setEnv___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__12___redArg___closed__2);
if (v_isShared_552_ == 0)
{
lean_ctor_set(v___x_551_, 5, v___x_553_);
lean_ctor_set(v___x_551_, 0, v_env_539_);
v___x_555_ = v___x_551_;
goto v_reusejp_554_;
}
else
{
lean_object* v_reuseFailAlloc_559_; 
v_reuseFailAlloc_559_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_559_, 0, v_env_539_);
lean_ctor_set(v_reuseFailAlloc_559_, 1, v_nextMacroScope_543_);
lean_ctor_set(v_reuseFailAlloc_559_, 2, v_ngen_544_);
lean_ctor_set(v_reuseFailAlloc_559_, 3, v_auxDeclNGen_545_);
lean_ctor_set(v_reuseFailAlloc_559_, 4, v_traceState_546_);
lean_ctor_set(v_reuseFailAlloc_559_, 5, v___x_553_);
lean_ctor_set(v_reuseFailAlloc_559_, 6, v_messages_547_);
lean_ctor_set(v_reuseFailAlloc_559_, 7, v_infoState_548_);
lean_ctor_set(v_reuseFailAlloc_559_, 8, v_snapshotTasks_549_);
v___x_555_ = v_reuseFailAlloc_559_;
goto v_reusejp_554_;
}
v_reusejp_554_:
{
lean_object* v___x_556_; lean_object* v___x_557_; lean_object* v___x_558_; 
v___x_556_ = lean_st_ref_set(v___y_540_, v___x_555_);
v___x_557_ = lean_box(0);
v___x_558_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_558_, 0, v___x_557_);
return v___x_558_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_setEnv___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__12___redArg___boxed(lean_object* v_env_563_, lean_object* v___y_564_, lean_object* v___y_565_){
_start:
{
lean_object* v_res_566_; 
v_res_566_ = lp_aesop_Lean_setEnv___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__12___redArg(v_env_563_, v___y_564_);
lean_dec(v___y_564_);
return v_res_566_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_modifyGetGlobalRuleSet___at___00__private_Aesop_Frontend_Extension_0__Aesop_Frontend_eraseGlobalRules_go___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0_spec__1_spec__3___redArg___lam__4(lean_object* v_snd_567_, lean_object* v_x_568_){
_start:
{
lean_object* v_toBaseRuleSet_569_; 
v_toBaseRuleSet_569_ = lean_ctor_get(v_snd_567_, 0);
lean_inc_ref(v_toBaseRuleSet_569_);
return v_toBaseRuleSet_569_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_modifyGetGlobalRuleSet___at___00__private_Aesop_Frontend_Extension_0__Aesop_Frontend_eraseGlobalRules_go___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0_spec__1_spec__3___redArg___lam__4___boxed(lean_object* v_snd_570_, lean_object* v_x_571_){
_start:
{
lean_object* v_res_572_; 
v_res_572_ = lp_aesop_Aesop_Frontend_modifyGetGlobalRuleSet___at___00__private_Aesop_Frontend_Extension_0__Aesop_Frontend_eraseGlobalRules_go___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0_spec__1_spec__3___redArg___lam__4(v_snd_570_, v_x_571_);
lean_dec_ref(v_x_571_);
lean_dec_ref(v_snd_570_);
return v_res_572_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_modifyGetGlobalRuleSet___at___00__private_Aesop_Frontend_Extension_0__Aesop_Frontend_eraseGlobalRules_go___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0_spec__1_spec__3___redArg(lean_object* v_rsName_576_, lean_object* v_f_577_, lean_object* v___y_578_, lean_object* v___y_579_){
_start:
{
lean_object* v___x_581_; 
v___x_581_ = lp_aesop_Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__9(v_rsName_576_, v___y_578_, v___y_579_);
if (lean_obj_tag(v___x_581_) == 0)
{
lean_object* v_a_582_; lean_object* v_snd_583_; lean_object* v_snd_584_; lean_object* v_snd_585_; lean_object* v_fst_586_; lean_object* v_fst_587_; lean_object* v_snd_588_; lean_object* v___x_589_; lean_object* v_ext_590_; lean_object* v_toEnvExtension_591_; lean_object* v_ext_592_; lean_object* v_toEnvExtension_593_; lean_object* v_ext_594_; lean_object* v_toEnvExtension_595_; lean_object* v_env_596_; lean_object* v_asyncMode_597_; lean_object* v_asyncMode_598_; lean_object* v_asyncMode_599_; lean_object* v___x_600_; lean_object* v_base_601_; lean_object* v___x_602_; lean_object* v___x_603_; lean_object* v_simpTheorems_604_; lean_object* v_simprocs_605_; lean_object* v_rs_606_; lean_object* v___x_607_; lean_object* v_fst_608_; lean_object* v_snd_609_; lean_object* v___f_610_; lean_object* v___f_611_; lean_object* v___f_612_; lean_object* v___f_613_; lean_object* v___f_614_; lean_object* v_env_615_; lean_object* v_env_616_; lean_object* v_env_617_; lean_object* v_env_618_; lean_object* v_env_619_; lean_object* v___x_620_; lean_object* v___x_622_; uint8_t v_isShared_623_; uint8_t v_isSharedCheck_627_; 
v_a_582_ = lean_ctor_get(v___x_581_, 0);
lean_inc(v_a_582_);
lean_dec_ref_known(v___x_581_, 1);
v_snd_583_ = lean_ctor_get(v_a_582_, 1);
v_snd_584_ = lean_ctor_get(v_snd_583_, 1);
lean_inc(v_snd_584_);
v_snd_585_ = lean_ctor_get(v_snd_584_, 1);
lean_inc(v_snd_585_);
v_fst_586_ = lean_ctor_get(v_a_582_, 0);
lean_inc_n(v_fst_586_, 2);
lean_dec(v_a_582_);
v_fst_587_ = lean_ctor_get(v_snd_584_, 0);
lean_inc_n(v_fst_587_, 2);
lean_dec(v_snd_584_);
v_snd_588_ = lean_ctor_get(v_snd_585_, 1);
lean_inc(v_snd_588_);
lean_dec(v_snd_585_);
v___x_589_ = lean_st_ref_get(v___y_579_);
v_ext_590_ = lean_ctor_get(v_fst_586_, 1);
v_toEnvExtension_591_ = lean_ctor_get(v_ext_590_, 0);
v_ext_592_ = lean_ctor_get(v_fst_587_, 1);
v_toEnvExtension_593_ = lean_ctor_get(v_ext_592_, 0);
v_ext_594_ = lean_ctor_get(v_snd_588_, 1);
v_toEnvExtension_595_ = lean_ctor_get(v_ext_594_, 0);
v_env_596_ = lean_ctor_get(v___x_589_, 0);
lean_inc_ref_n(v_env_596_, 4);
lean_dec(v___x_589_);
v_asyncMode_597_ = lean_ctor_get(v_toEnvExtension_591_, 2);
v_asyncMode_598_ = lean_ctor_get(v_toEnvExtension_593_, 2);
v_asyncMode_599_ = lean_ctor_get(v_toEnvExtension_595_, 2);
v___x_600_ = lp_aesop_Aesop_instInhabitedBaseRuleSet_default;
v_base_601_ = l_Lean_ScopedEnvExtension_getState___redArg(v___x_600_, v_fst_586_, v_env_596_, v_asyncMode_597_);
v___x_602_ = l_Lean_Meta_instInhabitedSimpTheorems_default;
v___x_603_ = l_Lean_Meta_Simp_instInhabitedSimprocs_default;
v_simpTheorems_604_ = l_Lean_ScopedEnvExtension_getState___redArg(v___x_602_, v_fst_587_, v_env_596_, v_asyncMode_598_);
v_simprocs_605_ = l_Lean_ScopedEnvExtension_getState___redArg(v___x_603_, v_snd_588_, v_env_596_, v_asyncMode_599_);
v_rs_606_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_rs_606_, 0, v_base_601_);
lean_ctor_set(v_rs_606_, 1, v_simpTheorems_604_);
lean_ctor_set(v_rs_606_, 2, v_simprocs_605_);
v___x_607_ = lean_apply_1(v_f_577_, v_rs_606_);
v_fst_608_ = lean_ctor_get(v___x_607_, 0);
lean_inc(v_fst_608_);
v_snd_609_ = lean_ctor_get(v___x_607_, 1);
lean_inc_n(v_snd_609_, 2);
lean_dec_ref(v___x_607_);
v___f_610_ = ((lean_object*)(lp_aesop_Aesop_Frontend_modifyGetGlobalRuleSet___at___00__private_Aesop_Frontend_Extension_0__Aesop_Frontend_eraseGlobalRules_go___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0_spec__1_spec__3___redArg___closed__0));
v___f_611_ = ((lean_object*)(lp_aesop_Aesop_Frontend_modifyGetGlobalRuleSet___at___00__private_Aesop_Frontend_Extension_0__Aesop_Frontend_eraseGlobalRules_go___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0_spec__1_spec__3___redArg___closed__1));
v___f_612_ = ((lean_object*)(lp_aesop_Aesop_Frontend_modifyGetGlobalRuleSet___at___00__private_Aesop_Frontend_Extension_0__Aesop_Frontend_eraseGlobalRules_go___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0_spec__1_spec__3___redArg___closed__2));
v___f_613_ = lean_alloc_closure((void*)(lp_aesop_Aesop_Frontend_modifyGetGlobalRuleSet___at___00__private_Aesop_Frontend_Extension_0__Aesop_Frontend_eraseGlobalRules_go___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0_spec__1_spec__3___redArg___lam__3___boxed), 2, 1);
lean_closure_set(v___f_613_, 0, v_snd_609_);
v___f_614_ = lean_alloc_closure((void*)(lp_aesop_Aesop_Frontend_modifyGetGlobalRuleSet___at___00__private_Aesop_Frontend_Extension_0__Aesop_Frontend_eraseGlobalRules_go___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0_spec__1_spec__3___redArg___lam__4___boxed), 2, 1);
lean_closure_set(v___f_614_, 0, v_snd_609_);
v_env_615_ = l_Lean_ScopedEnvExtension_modifyState___redArg(v_fst_586_, v_env_596_, v___f_612_);
v_env_616_ = l_Lean_ScopedEnvExtension_modifyState___redArg(v_fst_587_, v_env_615_, v___f_611_);
v_env_617_ = l_Lean_ScopedEnvExtension_modifyState___redArg(v_snd_588_, v_env_616_, v___f_610_);
v_env_618_ = l_Lean_ScopedEnvExtension_modifyState___redArg(v_fst_586_, v_env_617_, v___f_614_);
v_env_619_ = l_Lean_ScopedEnvExtension_modifyState___redArg(v_fst_587_, v_env_618_, v___f_613_);
v___x_620_ = lp_aesop_Lean_setEnv___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__12___redArg(v_env_619_, v___y_579_);
v_isSharedCheck_627_ = !lean_is_exclusive(v___x_620_);
if (v_isSharedCheck_627_ == 0)
{
lean_object* v_unused_628_; 
v_unused_628_ = lean_ctor_get(v___x_620_, 0);
lean_dec(v_unused_628_);
v___x_622_ = v___x_620_;
v_isShared_623_ = v_isSharedCheck_627_;
goto v_resetjp_621_;
}
else
{
lean_dec(v___x_620_);
v___x_622_ = lean_box(0);
v_isShared_623_ = v_isSharedCheck_627_;
goto v_resetjp_621_;
}
v_resetjp_621_:
{
lean_object* v___x_625_; 
if (v_isShared_623_ == 0)
{
lean_ctor_set(v___x_622_, 0, v_fst_608_);
v___x_625_ = v___x_622_;
goto v_reusejp_624_;
}
else
{
lean_object* v_reuseFailAlloc_626_; 
v_reuseFailAlloc_626_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_626_, 0, v_fst_608_);
v___x_625_ = v_reuseFailAlloc_626_;
goto v_reusejp_624_;
}
v_reusejp_624_:
{
return v___x_625_;
}
}
}
else
{
lean_object* v_a_629_; lean_object* v___x_631_; uint8_t v_isShared_632_; uint8_t v_isSharedCheck_636_; 
lean_dec_ref(v_f_577_);
v_a_629_ = lean_ctor_get(v___x_581_, 0);
v_isSharedCheck_636_ = !lean_is_exclusive(v___x_581_);
if (v_isSharedCheck_636_ == 0)
{
v___x_631_ = v___x_581_;
v_isShared_632_ = v_isSharedCheck_636_;
goto v_resetjp_630_;
}
else
{
lean_inc(v_a_629_);
lean_dec(v___x_581_);
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
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_modifyGetGlobalRuleSet___at___00__private_Aesop_Frontend_Extension_0__Aesop_Frontend_eraseGlobalRules_go___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0_spec__1_spec__3___redArg___boxed(lean_object* v_rsName_637_, lean_object* v_f_638_, lean_object* v___y_639_, lean_object* v___y_640_, lean_object* v___y_641_){
_start:
{
lean_object* v_res_642_; 
v_res_642_ = lp_aesop_Aesop_Frontend_modifyGetGlobalRuleSet___at___00__private_Aesop_Frontend_Extension_0__Aesop_Frontend_eraseGlobalRules_go___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0_spec__1_spec__3___redArg(v_rsName_637_, v_f_638_, v___y_639_, v___y_640_);
lean_dec(v___y_640_);
lean_dec_ref(v___y_639_);
return v_res_642_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Frontend_Extension_0__Aesop_Frontend_eraseGlobalRules_go___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0_spec__1___lam__0(lean_object* v_rf_643_, uint8_t v_anyErased_644_, lean_object* v_rs_645_){
_start:
{
lean_object* v___x_646_; 
v___x_646_ = lp_aesop_Aesop_GlobalRuleSet_erase(v_rs_645_, v_rf_643_);
if (v_anyErased_644_ == 0)
{
lean_object* v_fst_647_; lean_object* v_snd_648_; lean_object* v___x_650_; uint8_t v_isShared_651_; uint8_t v_isSharedCheck_655_; 
v_fst_647_ = lean_ctor_get(v___x_646_, 0);
v_snd_648_ = lean_ctor_get(v___x_646_, 1);
v_isSharedCheck_655_ = !lean_is_exclusive(v___x_646_);
if (v_isSharedCheck_655_ == 0)
{
v___x_650_ = v___x_646_;
v_isShared_651_ = v_isSharedCheck_655_;
goto v_resetjp_649_;
}
else
{
lean_inc(v_snd_648_);
lean_inc(v_fst_647_);
lean_dec(v___x_646_);
v___x_650_ = lean_box(0);
v_isShared_651_ = v_isSharedCheck_655_;
goto v_resetjp_649_;
}
v_resetjp_649_:
{
lean_object* v___x_653_; 
if (v_isShared_651_ == 0)
{
lean_ctor_set(v___x_650_, 1, v_fst_647_);
lean_ctor_set(v___x_650_, 0, v_snd_648_);
v___x_653_ = v___x_650_;
goto v_reusejp_652_;
}
else
{
lean_object* v_reuseFailAlloc_654_; 
v_reuseFailAlloc_654_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_654_, 0, v_snd_648_);
lean_ctor_set(v_reuseFailAlloc_654_, 1, v_fst_647_);
v___x_653_ = v_reuseFailAlloc_654_;
goto v_reusejp_652_;
}
v_reusejp_652_:
{
return v___x_653_;
}
}
}
else
{
lean_object* v_fst_656_; lean_object* v___x_658_; uint8_t v_isShared_659_; uint8_t v_isSharedCheck_664_; 
v_fst_656_ = lean_ctor_get(v___x_646_, 0);
v_isSharedCheck_664_ = !lean_is_exclusive(v___x_646_);
if (v_isSharedCheck_664_ == 0)
{
lean_object* v_unused_665_; 
v_unused_665_ = lean_ctor_get(v___x_646_, 1);
lean_dec(v_unused_665_);
v___x_658_ = v___x_646_;
v_isShared_659_ = v_isSharedCheck_664_;
goto v_resetjp_657_;
}
else
{
lean_inc(v_fst_656_);
lean_dec(v___x_646_);
v___x_658_ = lean_box(0);
v_isShared_659_ = v_isSharedCheck_664_;
goto v_resetjp_657_;
}
v_resetjp_657_:
{
lean_object* v___x_660_; lean_object* v___x_662_; 
v___x_660_ = lean_box(v_anyErased_644_);
if (v_isShared_659_ == 0)
{
lean_ctor_set(v___x_658_, 1, v_fst_656_);
lean_ctor_set(v___x_658_, 0, v___x_660_);
v___x_662_ = v___x_658_;
goto v_reusejp_661_;
}
else
{
lean_object* v_reuseFailAlloc_663_; 
v_reuseFailAlloc_663_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_663_, 0, v___x_660_);
lean_ctor_set(v_reuseFailAlloc_663_, 1, v_fst_656_);
v___x_662_ = v_reuseFailAlloc_663_;
goto v_reusejp_661_;
}
v_reusejp_661_:
{
return v___x_662_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Frontend_Extension_0__Aesop_Frontend_eraseGlobalRules_go___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0_spec__1___lam__0___boxed(lean_object* v_rf_666_, lean_object* v_anyErased_667_, lean_object* v_rs_668_){
_start:
{
uint8_t v_anyErased_boxed_669_; lean_object* v_res_670_; 
v_anyErased_boxed_669_ = lean_unbox(v_anyErased_667_);
v_res_670_ = lp_aesop___private_Aesop_Frontend_Extension_0__Aesop_Frontend_eraseGlobalRules_go___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0_spec__1___lam__0(v_rf_666_, v_anyErased_boxed_669_, v_rs_668_);
return v_res_670_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Frontend_Extension_0__Aesop_Frontend_eraseGlobalRules_go___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0_spec__1(lean_object* v_rf_671_, uint8_t v_anyErased_672_, lean_object* v_rsName_673_, lean_object* v___y_674_, lean_object* v___y_675_){
_start:
{
lean_object* v___x_677_; lean_object* v___f_678_; lean_object* v___x_679_; 
v___x_677_ = lean_box(v_anyErased_672_);
v___f_678_ = lean_alloc_closure((void*)(lp_aesop___private_Aesop_Frontend_Extension_0__Aesop_Frontend_eraseGlobalRules_go___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0_spec__1___lam__0___boxed), 3, 2);
lean_closure_set(v___f_678_, 0, v_rf_671_);
lean_closure_set(v___f_678_, 1, v___x_677_);
v___x_679_ = lp_aesop_Aesop_Frontend_modifyGetGlobalRuleSet___at___00__private_Aesop_Frontend_Extension_0__Aesop_Frontend_eraseGlobalRules_go___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0_spec__1_spec__3___redArg(v_rsName_673_, v___f_678_, v___y_674_, v___y_675_);
return v___x_679_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Frontend_Extension_0__Aesop_Frontend_eraseGlobalRules_go___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0_spec__1___boxed(lean_object* v_rf_680_, lean_object* v_anyErased_681_, lean_object* v_rsName_682_, lean_object* v___y_683_, lean_object* v___y_684_, lean_object* v___y_685_){
_start:
{
uint8_t v_anyErased_boxed_686_; lean_object* v_res_687_; 
v_anyErased_boxed_686_ = lean_unbox(v_anyErased_681_);
v_res_687_ = lp_aesop___private_Aesop_Frontend_Extension_0__Aesop_Frontend_eraseGlobalRules_go___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0_spec__1(v_rf_680_, v_anyErased_boxed_686_, v_rsName_682_, v___y_683_, v___y_684_);
lean_dec(v___y_684_);
lean_dec_ref(v___y_683_);
return v_res_687_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0_spec__6(lean_object* v_rf_688_, lean_object* v_as_689_, size_t v_i_690_, size_t v_stop_691_, uint8_t v_b_692_, lean_object* v___y_693_, lean_object* v___y_694_){
_start:
{
uint8_t v___x_696_; 
v___x_696_ = lean_usize_dec_eq(v_i_690_, v_stop_691_);
if (v___x_696_ == 0)
{
lean_object* v___x_697_; lean_object* v___x_698_; 
v___x_697_ = lean_array_uget_borrowed(v_as_689_, v_i_690_);
lean_inc(v___x_697_);
lean_inc_ref(v_rf_688_);
v___x_698_ = lp_aesop___private_Aesop_Frontend_Extension_0__Aesop_Frontend_eraseGlobalRules_go___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0_spec__1(v_rf_688_, v_b_692_, v___x_697_, v___y_693_, v___y_694_);
if (lean_obj_tag(v___x_698_) == 0)
{
lean_object* v_a_699_; size_t v___x_700_; size_t v___x_701_; uint8_t v___x_702_; 
v_a_699_ = lean_ctor_get(v___x_698_, 0);
lean_inc(v_a_699_);
lean_dec_ref_known(v___x_698_, 1);
v___x_700_ = ((size_t)1ULL);
v___x_701_ = lean_usize_add(v_i_690_, v___x_700_);
v___x_702_ = lean_unbox(v_a_699_);
lean_dec(v_a_699_);
v_i_690_ = v___x_701_;
v_b_692_ = v___x_702_;
goto _start;
}
else
{
lean_dec_ref(v_rf_688_);
return v___x_698_;
}
}
else
{
lean_object* v___x_704_; lean_object* v___x_705_; 
lean_dec_ref(v_rf_688_);
v___x_704_ = lean_box(v_b_692_);
v___x_705_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_705_, 0, v___x_704_);
return v___x_705_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0_spec__6___boxed(lean_object* v_rf_706_, lean_object* v_as_707_, lean_object* v_i_708_, lean_object* v_stop_709_, lean_object* v_b_710_, lean_object* v___y_711_, lean_object* v___y_712_, lean_object* v___y_713_){
_start:
{
size_t v_i_boxed_714_; size_t v_stop_boxed_715_; uint8_t v_b_boxed_716_; lean_object* v_res_717_; 
v_i_boxed_714_ = lean_unbox_usize(v_i_708_);
lean_dec(v_i_708_);
v_stop_boxed_715_ = lean_unbox_usize(v_stop_709_);
lean_dec(v_stop_709_);
v_b_boxed_716_ = lean_unbox(v_b_710_);
v_res_717_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0_spec__6(v_rf_706_, v_as_707_, v_i_boxed_714_, v_stop_boxed_715_, v_b_boxed_716_, v___y_711_, v___y_712_);
lean_dec(v___y_712_);
lean_dec_ref(v___y_711_);
lean_dec_ref(v_as_707_);
return v_res_717_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0_spec__2(lean_object* v_rf_718_, uint8_t v_x_719_, lean_object* v_x_720_, lean_object* v___y_721_, lean_object* v___y_722_){
_start:
{
if (lean_obj_tag(v_x_720_) == 0)
{
lean_object* v___x_724_; lean_object* v___x_725_; 
lean_dec_ref(v_rf_718_);
v___x_724_ = lean_box(v_x_719_);
v___x_725_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_725_, 0, v___x_724_);
return v___x_725_;
}
else
{
lean_object* v_key_726_; lean_object* v_tail_727_; lean_object* v___x_728_; 
v_key_726_ = lean_ctor_get(v_x_720_, 0);
lean_inc(v_key_726_);
v_tail_727_ = lean_ctor_get(v_x_720_, 2);
lean_inc(v_tail_727_);
lean_dec_ref_known(v_x_720_, 3);
lean_inc_ref(v_rf_718_);
v___x_728_ = lp_aesop___private_Aesop_Frontend_Extension_0__Aesop_Frontend_eraseGlobalRules_go___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0_spec__1(v_rf_718_, v_x_719_, v_key_726_, v___y_721_, v___y_722_);
if (lean_obj_tag(v___x_728_) == 0)
{
lean_object* v_a_729_; uint8_t v___x_730_; 
v_a_729_ = lean_ctor_get(v___x_728_, 0);
lean_inc(v_a_729_);
lean_dec_ref_known(v___x_728_, 1);
v___x_730_ = lean_unbox(v_a_729_);
lean_dec(v_a_729_);
v_x_719_ = v___x_730_;
v_x_720_ = v_tail_727_;
goto _start;
}
else
{
lean_dec(v_tail_727_);
lean_dec_ref(v_rf_718_);
return v___x_728_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0_spec__2___boxed(lean_object* v_rf_732_, lean_object* v_x_733_, lean_object* v_x_734_, lean_object* v___y_735_, lean_object* v___y_736_, lean_object* v___y_737_){
_start:
{
uint8_t v_x_10023__boxed_738_; lean_object* v_res_739_; 
v_x_10023__boxed_738_ = lean_unbox(v_x_733_);
v_res_739_ = lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0_spec__2(v_rf_732_, v_x_10023__boxed_738_, v_x_734_, v___y_735_, v___y_736_);
lean_dec(v___y_736_);
lean_dec_ref(v___y_735_);
return v_res_739_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0_spec__3(lean_object* v_rf_740_, lean_object* v_as_741_, size_t v_i_742_, size_t v_stop_743_, uint8_t v_b_744_, lean_object* v___y_745_, lean_object* v___y_746_){
_start:
{
uint8_t v___x_748_; 
v___x_748_ = lean_usize_dec_eq(v_i_742_, v_stop_743_);
if (v___x_748_ == 0)
{
lean_object* v___x_749_; lean_object* v___x_750_; 
v___x_749_ = lean_array_uget_borrowed(v_as_741_, v_i_742_);
lean_inc(v___x_749_);
lean_inc_ref(v_rf_740_);
v___x_750_ = lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0_spec__2(v_rf_740_, v_b_744_, v___x_749_, v___y_745_, v___y_746_);
if (lean_obj_tag(v___x_750_) == 0)
{
lean_object* v_a_751_; size_t v___x_752_; size_t v___x_753_; uint8_t v___x_754_; 
v_a_751_ = lean_ctor_get(v___x_750_, 0);
lean_inc(v_a_751_);
lean_dec_ref_known(v___x_750_, 1);
v___x_752_ = ((size_t)1ULL);
v___x_753_ = lean_usize_add(v_i_742_, v___x_752_);
v___x_754_ = lean_unbox(v_a_751_);
lean_dec(v_a_751_);
v_i_742_ = v___x_753_;
v_b_744_ = v___x_754_;
goto _start;
}
else
{
lean_dec_ref(v_rf_740_);
return v___x_750_;
}
}
else
{
lean_object* v___x_756_; lean_object* v___x_757_; 
lean_dec_ref(v_rf_740_);
v___x_756_ = lean_box(v_b_744_);
v___x_757_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_757_, 0, v___x_756_);
return v___x_757_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0_spec__3___boxed(lean_object* v_rf_758_, lean_object* v_as_759_, lean_object* v_i_760_, lean_object* v_stop_761_, lean_object* v_b_762_, lean_object* v___y_763_, lean_object* v___y_764_, lean_object* v___y_765_){
_start:
{
size_t v_i_boxed_766_; size_t v_stop_boxed_767_; uint8_t v_b_boxed_768_; lean_object* v_res_769_; 
v_i_boxed_766_ = lean_unbox_usize(v_i_760_);
lean_dec(v_i_760_);
v_stop_boxed_767_ = lean_unbox_usize(v_stop_761_);
lean_dec(v_stop_761_);
v_b_boxed_768_ = lean_unbox(v_b_762_);
v_res_769_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0_spec__3(v_rf_758_, v_as_759_, v_i_boxed_766_, v_stop_boxed_767_, v_b_boxed_768_, v___y_763_, v___y_764_);
lean_dec(v___y_764_);
lean_dec_ref(v___y_763_);
lean_dec_ref(v_as_759_);
return v_res_769_;
}
}
LEAN_EXPORT lean_object* lp_aesop_List_mapTR_loop___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0_spec__5(lean_object* v_a_770_, lean_object* v_a_771_){
_start:
{
if (lean_obj_tag(v_a_770_) == 0)
{
lean_object* v___x_772_; 
v___x_772_ = l_List_reverse___redArg(v_a_771_);
return v___x_772_;
}
else
{
lean_object* v_head_773_; lean_object* v_tail_774_; lean_object* v___x_776_; uint8_t v_isShared_777_; uint8_t v_isSharedCheck_783_; 
v_head_773_ = lean_ctor_get(v_a_770_, 0);
v_tail_774_ = lean_ctor_get(v_a_770_, 1);
v_isSharedCheck_783_ = !lean_is_exclusive(v_a_770_);
if (v_isSharedCheck_783_ == 0)
{
v___x_776_ = v_a_770_;
v_isShared_777_ = v_isSharedCheck_783_;
goto v_resetjp_775_;
}
else
{
lean_inc(v_tail_774_);
lean_inc(v_head_773_);
lean_dec(v_a_770_);
v___x_776_ = lean_box(0);
v_isShared_777_ = v_isSharedCheck_783_;
goto v_resetjp_775_;
}
v_resetjp_775_:
{
lean_object* v___x_778_; lean_object* v___x_780_; 
v___x_778_ = l_Lean_stringToMessageData(v_head_773_);
if (v_isShared_777_ == 0)
{
lean_ctor_set(v___x_776_, 1, v_a_771_);
lean_ctor_set(v___x_776_, 0, v___x_778_);
v___x_780_ = v___x_776_;
goto v_reusejp_779_;
}
else
{
lean_object* v_reuseFailAlloc_782_; 
v_reuseFailAlloc_782_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_782_, 0, v___x_778_);
lean_ctor_set(v_reuseFailAlloc_782_, 1, v_a_771_);
v___x_780_ = v_reuseFailAlloc_782_;
goto v_reusejp_779_;
}
v_reusejp_779_:
{
v_a_770_ = v_tail_774_;
v_a_771_ = v___x_780_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0_spec__4(uint8_t v_checkExists_784_, size_t v_sz_785_, size_t v_i_786_, lean_object* v_bs_787_){
_start:
{
uint8_t v___x_788_; 
v___x_788_ = lean_usize_dec_lt(v_i_786_, v_sz_785_);
if (v___x_788_ == 0)
{
return v_bs_787_;
}
else
{
lean_object* v_v_789_; lean_object* v___x_790_; lean_object* v_bs_x27_791_; lean_object* v___x_792_; size_t v___x_793_; size_t v___x_794_; lean_object* v___x_795_; 
v_v_789_ = lean_array_uget(v_bs_787_, v_i_786_);
v___x_790_ = lean_unsigned_to_nat(0u);
v_bs_x27_791_ = lean_array_uset(v_bs_787_, v_i_786_, v___x_790_);
v___x_792_ = l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(v_v_789_, v_checkExists_784_);
v___x_793_ = ((size_t)1ULL);
v___x_794_ = lean_usize_add(v_i_786_, v___x_793_);
v___x_795_ = lean_array_uset(v_bs_x27_791_, v_i_786_, v___x_792_);
v_i_786_ = v___x_794_;
v_bs_787_ = v___x_795_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0_spec__4___boxed(lean_object* v_checkExists_797_, lean_object* v_sz_798_, lean_object* v_i_799_, lean_object* v_bs_800_){
_start:
{
uint8_t v_checkExists_boxed_801_; size_t v_sz_boxed_802_; size_t v_i_boxed_803_; lean_object* v_res_804_; 
v_checkExists_boxed_801_ = lean_unbox(v_checkExists_797_);
v_sz_boxed_802_ = lean_unbox_usize(v_sz_798_);
lean_dec(v_sz_798_);
v_i_boxed_803_ = lean_unbox_usize(v_i_799_);
lean_dec(v_i_799_);
v_res_804_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0_spec__4(v_checkExists_boxed_801_, v_sz_boxed_802_, v_i_boxed_803_, v_bs_800_);
return v_res_804_;
}
}
static lean_object* _init_lp_aesop_Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0___closed__1(void){
_start:
{
lean_object* v___x_806_; lean_object* v___x_807_; 
v___x_806_ = ((lean_object*)(lp_aesop_Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0___closed__0));
v___x_807_ = l_Lean_stringToMessageData(v___x_806_);
return v___x_807_;
}
}
static lean_object* _init_lp_aesop_Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0___closed__3(void){
_start:
{
lean_object* v___x_809_; lean_object* v___x_810_; 
v___x_809_ = ((lean_object*)(lp_aesop_Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0___closed__2));
v___x_810_ = l_Lean_stringToMessageData(v___x_809_);
return v___x_810_;
}
}
static lean_object* _init_lp_aesop_Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0___closed__5(void){
_start:
{
lean_object* v___x_812_; lean_object* v___x_813_; 
v___x_812_ = ((lean_object*)(lp_aesop_Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0___closed__4));
v___x_813_ = l_Lean_stringToMessageData(v___x_812_);
return v___x_813_;
}
}
static lean_object* _init_lp_aesop_Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0___closed__7(void){
_start:
{
lean_object* v___x_815_; lean_object* v___x_816_; 
v___x_815_ = ((lean_object*)(lp_aesop_Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0___closed__6));
v___x_816_ = l_Lean_stringToMessageData(v___x_815_);
return v___x_816_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0(lean_object* v_rsf_817_, lean_object* v_rf_818_, uint8_t v_checkExists_819_, lean_object* v___y_820_, lean_object* v___y_821_){
_start:
{
uint8_t v_anyErased_830_; lean_object* v___y_831_; lean_object* v___y_832_; lean_object* v___x_840_; 
v___x_840_ = lp_aesop_Aesop_RuleSetNameFilter_matchedRuleSetNames(v_rsf_817_);
if (lean_obj_tag(v___x_840_) == 0)
{
lean_object* v___x_841_; 
v___x_841_ = lp_aesop_Aesop_getDeclaredRuleSets();
if (lean_obj_tag(v___x_841_) == 0)
{
lean_object* v_a_842_; lean_object* v___x_844_; uint8_t v_isShared_845_; uint8_t v_isSharedCheck_896_; 
v_a_842_ = lean_ctor_get(v___x_841_, 0);
v_isSharedCheck_896_ = !lean_is_exclusive(v___x_841_);
if (v_isSharedCheck_896_ == 0)
{
v___x_844_ = v___x_841_;
v_isShared_845_ = v_isSharedCheck_896_;
goto v_resetjp_843_;
}
else
{
lean_inc(v_a_842_);
lean_dec(v___x_841_);
v___x_844_ = lean_box(0);
v_isShared_845_ = v_isSharedCheck_896_;
goto v_resetjp_843_;
}
v_resetjp_843_:
{
lean_object* v_buckets_846_; lean_object* v___x_848_; uint8_t v_isShared_849_; uint8_t v_isSharedCheck_894_; 
v_buckets_846_ = lean_ctor_get(v_a_842_, 1);
v_isSharedCheck_894_ = !lean_is_exclusive(v_a_842_);
if (v_isSharedCheck_894_ == 0)
{
lean_object* v_unused_895_; 
v_unused_895_ = lean_ctor_get(v_a_842_, 0);
lean_dec(v_unused_895_);
v___x_848_ = v_a_842_;
v_isShared_849_ = v_isSharedCheck_894_;
goto v_resetjp_847_;
}
else
{
lean_inc(v_buckets_846_);
lean_dec(v_a_842_);
v___x_848_ = lean_box(0);
v_isShared_849_ = v_isSharedCheck_894_;
goto v_resetjp_847_;
}
v_resetjp_847_:
{
lean_object* v___x_850_; lean_object* v___x_851_; uint8_t v___x_852_; 
v___x_850_ = lean_unsigned_to_nat(0u);
v___x_851_ = lean_array_get_size(v_buckets_846_);
v___x_852_ = lean_nat_dec_lt(v___x_850_, v___x_851_);
if (v___x_852_ == 0)
{
lean_dec_ref(v_buckets_846_);
if (v_checkExists_819_ == 0)
{
lean_object* v___x_853_; lean_object* v___x_855_; 
lean_del_object(v___x_848_);
lean_dec_ref(v_rf_818_);
v___x_853_ = lean_box(0);
if (v_isShared_845_ == 0)
{
lean_ctor_set(v___x_844_, 0, v___x_853_);
v___x_855_ = v___x_844_;
goto v_reusejp_854_;
}
else
{
lean_object* v_reuseFailAlloc_856_; 
v_reuseFailAlloc_856_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_856_, 0, v___x_853_);
v___x_855_ = v_reuseFailAlloc_856_;
goto v_reusejp_854_;
}
v_reusejp_854_:
{
return v___x_855_;
}
}
else
{
lean_object* v_name_857_; lean_object* v___x_858_; lean_object* v___x_859_; lean_object* v___x_861_; 
lean_del_object(v___x_844_);
v_name_857_ = lean_ctor_get(v_rf_818_, 0);
lean_inc(v_name_857_);
lean_dec_ref(v_rf_818_);
v___x_858_ = lean_obj_once(&lp_aesop_Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0___closed__1, &lp_aesop_Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0___closed__1_once, _init_lp_aesop_Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0___closed__1);
v___x_859_ = l_Lean_MessageData_ofName(v_name_857_);
if (v_isShared_849_ == 0)
{
lean_ctor_set_tag(v___x_848_, 7);
lean_ctor_set(v___x_848_, 1, v___x_859_);
lean_ctor_set(v___x_848_, 0, v___x_858_);
v___x_861_ = v___x_848_;
goto v_reusejp_860_;
}
else
{
lean_object* v_reuseFailAlloc_865_; 
v_reuseFailAlloc_865_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_865_, 0, v___x_858_);
lean_ctor_set(v_reuseFailAlloc_865_, 1, v___x_859_);
v___x_861_ = v_reuseFailAlloc_865_;
goto v_reusejp_860_;
}
v_reusejp_860_:
{
lean_object* v___x_862_; lean_object* v___x_863_; lean_object* v___x_864_; 
v___x_862_ = lean_obj_once(&lp_aesop_Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0___closed__3, &lp_aesop_Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0___closed__3_once, _init_lp_aesop_Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0___closed__3);
v___x_863_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_863_, 0, v___x_861_);
lean_ctor_set(v___x_863_, 1, v___x_862_);
v___x_864_ = lp_aesop_Lean_throwError___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0_spec__0___redArg(v___x_863_, v___y_820_, v___y_821_);
return v___x_864_;
}
}
}
else
{
uint8_t v___x_866_; uint8_t v___x_867_; 
lean_del_object(v___x_848_);
lean_del_object(v___x_844_);
v___x_866_ = 0;
v___x_867_ = lean_nat_dec_le(v___x_851_, v___x_851_);
if (v___x_867_ == 0)
{
if (v___x_852_ == 0)
{
lean_dec_ref(v_buckets_846_);
v_anyErased_830_ = v___x_866_;
v___y_831_ = v___y_820_;
v___y_832_ = v___y_821_;
goto v___jp_829_;
}
else
{
size_t v___x_868_; size_t v___x_869_; lean_object* v___x_870_; 
v___x_868_ = ((size_t)0ULL);
v___x_869_ = lean_usize_of_nat(v___x_851_);
lean_inc_ref(v_rf_818_);
v___x_870_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0_spec__3(v_rf_818_, v_buckets_846_, v___x_868_, v___x_869_, v___x_866_, v___y_820_, v___y_821_);
lean_dec_ref(v_buckets_846_);
if (lean_obj_tag(v___x_870_) == 0)
{
lean_object* v_a_871_; uint8_t v___x_872_; 
v_a_871_ = lean_ctor_get(v___x_870_, 0);
lean_inc(v_a_871_);
lean_dec_ref_known(v___x_870_, 1);
v___x_872_ = lean_unbox(v_a_871_);
lean_dec(v_a_871_);
v_anyErased_830_ = v___x_872_;
v___y_831_ = v___y_820_;
v___y_832_ = v___y_821_;
goto v___jp_829_;
}
else
{
lean_object* v_a_873_; lean_object* v___x_875_; uint8_t v_isShared_876_; uint8_t v_isSharedCheck_880_; 
lean_dec_ref(v_rf_818_);
v_a_873_ = lean_ctor_get(v___x_870_, 0);
v_isSharedCheck_880_ = !lean_is_exclusive(v___x_870_);
if (v_isSharedCheck_880_ == 0)
{
v___x_875_ = v___x_870_;
v_isShared_876_ = v_isSharedCheck_880_;
goto v_resetjp_874_;
}
else
{
lean_inc(v_a_873_);
lean_dec(v___x_870_);
v___x_875_ = lean_box(0);
v_isShared_876_ = v_isSharedCheck_880_;
goto v_resetjp_874_;
}
v_resetjp_874_:
{
lean_object* v___x_878_; 
if (v_isShared_876_ == 0)
{
v___x_878_ = v___x_875_;
goto v_reusejp_877_;
}
else
{
lean_object* v_reuseFailAlloc_879_; 
v_reuseFailAlloc_879_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_879_, 0, v_a_873_);
v___x_878_ = v_reuseFailAlloc_879_;
goto v_reusejp_877_;
}
v_reusejp_877_:
{
return v___x_878_;
}
}
}
}
}
else
{
size_t v___x_881_; size_t v___x_882_; lean_object* v___x_883_; 
v___x_881_ = ((size_t)0ULL);
v___x_882_ = lean_usize_of_nat(v___x_851_);
lean_inc_ref(v_rf_818_);
v___x_883_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0_spec__3(v_rf_818_, v_buckets_846_, v___x_881_, v___x_882_, v___x_866_, v___y_820_, v___y_821_);
lean_dec_ref(v_buckets_846_);
if (lean_obj_tag(v___x_883_) == 0)
{
lean_object* v_a_884_; uint8_t v___x_885_; 
v_a_884_ = lean_ctor_get(v___x_883_, 0);
lean_inc(v_a_884_);
lean_dec_ref_known(v___x_883_, 1);
v___x_885_ = lean_unbox(v_a_884_);
lean_dec(v_a_884_);
v_anyErased_830_ = v___x_885_;
v___y_831_ = v___y_820_;
v___y_832_ = v___y_821_;
goto v___jp_829_;
}
else
{
lean_object* v_a_886_; lean_object* v___x_888_; uint8_t v_isShared_889_; uint8_t v_isSharedCheck_893_; 
lean_dec_ref(v_rf_818_);
v_a_886_ = lean_ctor_get(v___x_883_, 0);
v_isSharedCheck_893_ = !lean_is_exclusive(v___x_883_);
if (v_isSharedCheck_893_ == 0)
{
v___x_888_ = v___x_883_;
v_isShared_889_ = v_isSharedCheck_893_;
goto v_resetjp_887_;
}
else
{
lean_inc(v_a_886_);
lean_dec(v___x_883_);
v___x_888_ = lean_box(0);
v_isShared_889_ = v_isSharedCheck_893_;
goto v_resetjp_887_;
}
v_resetjp_887_:
{
lean_object* v___x_891_; 
if (v_isShared_889_ == 0)
{
v___x_891_ = v___x_888_;
goto v_reusejp_890_;
}
else
{
lean_object* v_reuseFailAlloc_892_; 
v_reuseFailAlloc_892_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_892_, 0, v_a_886_);
v___x_891_ = v_reuseFailAlloc_892_;
goto v_reusejp_890_;
}
v_reusejp_890_:
{
return v___x_891_;
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
lean_object* v_a_897_; lean_object* v___x_899_; uint8_t v_isShared_900_; uint8_t v_isSharedCheck_909_; 
lean_dec_ref(v_rf_818_);
v_a_897_ = lean_ctor_get(v___x_841_, 0);
v_isSharedCheck_909_ = !lean_is_exclusive(v___x_841_);
if (v_isSharedCheck_909_ == 0)
{
v___x_899_ = v___x_841_;
v_isShared_900_ = v_isSharedCheck_909_;
goto v_resetjp_898_;
}
else
{
lean_inc(v_a_897_);
lean_dec(v___x_841_);
v___x_899_ = lean_box(0);
v_isShared_900_ = v_isSharedCheck_909_;
goto v_resetjp_898_;
}
v_resetjp_898_:
{
lean_object* v_ref_901_; lean_object* v___x_902_; lean_object* v___x_903_; lean_object* v___x_904_; lean_object* v___x_905_; lean_object* v___x_907_; 
v_ref_901_ = lean_ctor_get(v___y_820_, 5);
v___x_902_ = lean_io_error_to_string(v_a_897_);
v___x_903_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_903_, 0, v___x_902_);
v___x_904_ = l_Lean_MessageData_ofFormat(v___x_903_);
lean_inc(v_ref_901_);
v___x_905_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_905_, 0, v_ref_901_);
lean_ctor_set(v___x_905_, 1, v___x_904_);
if (v_isShared_900_ == 0)
{
lean_ctor_set(v___x_899_, 0, v___x_905_);
v___x_907_ = v___x_899_;
goto v_reusejp_906_;
}
else
{
lean_object* v_reuseFailAlloc_908_; 
v_reuseFailAlloc_908_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_908_, 0, v___x_905_);
v___x_907_ = v_reuseFailAlloc_908_;
goto v_reusejp_906_;
}
v_reusejp_906_:
{
return v___x_907_;
}
}
}
}
else
{
lean_object* v_val_910_; lean_object* v___x_912_; uint8_t v_isShared_913_; uint8_t v_isSharedCheck_987_; 
v_val_910_ = lean_ctor_get(v___x_840_, 0);
v_isSharedCheck_987_ = !lean_is_exclusive(v___x_840_);
if (v_isSharedCheck_987_ == 0)
{
v___x_912_ = v___x_840_;
v_isShared_913_ = v_isSharedCheck_987_;
goto v_resetjp_911_;
}
else
{
lean_inc(v_val_910_);
lean_dec(v___x_840_);
v___x_912_ = lean_box(0);
v_isShared_913_ = v_isSharedCheck_987_;
goto v_resetjp_911_;
}
v_resetjp_911_:
{
uint8_t v_anyErased_915_; lean_object* v___y_916_; lean_object* v___y_917_; lean_object* v___x_935_; lean_object* v___x_936_; uint8_t v___x_937_; 
v___x_935_ = lean_unsigned_to_nat(0u);
v___x_936_ = lean_array_get_size(v_val_910_);
v___x_937_ = lean_nat_dec_lt(v___x_935_, v___x_936_);
if (v___x_937_ == 0)
{
if (v_checkExists_819_ == 0)
{
lean_object* v___x_938_; lean_object* v___x_940_; 
lean_dec(v_val_910_);
lean_dec_ref(v_rf_818_);
v___x_938_ = lean_box(0);
if (v_isShared_913_ == 0)
{
lean_ctor_set_tag(v___x_912_, 0);
lean_ctor_set(v___x_912_, 0, v___x_938_);
v___x_940_ = v___x_912_;
goto v_reusejp_939_;
}
else
{
lean_object* v_reuseFailAlloc_941_; 
v_reuseFailAlloc_941_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_941_, 0, v___x_938_);
v___x_940_ = v_reuseFailAlloc_941_;
goto v_reusejp_939_;
}
v_reusejp_939_:
{
return v___x_940_;
}
}
else
{
lean_object* v_name_942_; lean_object* v___x_943_; lean_object* v___x_944_; lean_object* v___x_945_; lean_object* v___x_946_; lean_object* v___x_947_; size_t v_sz_948_; size_t v___x_949_; lean_object* v___x_950_; lean_object* v___x_951_; lean_object* v___x_952_; lean_object* v___x_953_; lean_object* v___x_954_; lean_object* v___x_955_; lean_object* v___x_956_; lean_object* v___x_957_; lean_object* v___x_958_; 
lean_del_object(v___x_912_);
v_name_942_ = lean_ctor_get(v_rf_818_, 0);
lean_inc(v_name_942_);
lean_dec_ref(v_rf_818_);
v___x_943_ = lean_obj_once(&lp_aesop_Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0___closed__1, &lp_aesop_Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0___closed__1_once, _init_lp_aesop_Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0___closed__1);
v___x_944_ = l_Lean_MessageData_ofName(v_name_942_);
v___x_945_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_945_, 0, v___x_943_);
lean_ctor_set(v___x_945_, 1, v___x_944_);
v___x_946_ = lean_obj_once(&lp_aesop_Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0___closed__5, &lp_aesop_Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0___closed__5_once, _init_lp_aesop_Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0___closed__5);
v___x_947_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_947_, 0, v___x_945_);
lean_ctor_set(v___x_947_, 1, v___x_946_);
v_sz_948_ = lean_array_size(v_val_910_);
v___x_949_ = ((size_t)0ULL);
v___x_950_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0_spec__4(v_checkExists_819_, v_sz_948_, v___x_949_, v_val_910_);
v___x_951_ = lean_array_to_list(v___x_950_);
v___x_952_ = lean_box(0);
v___x_953_ = lp_aesop_List_mapTR_loop___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0_spec__5(v___x_951_, v___x_952_);
v___x_954_ = l_Lean_MessageData_ofList(v___x_953_);
v___x_955_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_955_, 0, v___x_947_);
lean_ctor_set(v___x_955_, 1, v___x_954_);
v___x_956_ = lean_obj_once(&lp_aesop_Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0___closed__7, &lp_aesop_Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0___closed__7_once, _init_lp_aesop_Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0___closed__7);
v___x_957_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_957_, 0, v___x_955_);
lean_ctor_set(v___x_957_, 1, v___x_956_);
v___x_958_ = lp_aesop_Lean_throwError___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0_spec__0___redArg(v___x_957_, v___y_820_, v___y_821_);
return v___x_958_;
}
}
else
{
uint8_t v___x_959_; uint8_t v___x_960_; 
lean_del_object(v___x_912_);
v___x_959_ = 0;
v___x_960_ = lean_nat_dec_le(v___x_936_, v___x_936_);
if (v___x_960_ == 0)
{
if (v___x_937_ == 0)
{
v_anyErased_915_ = v___x_959_;
v___y_916_ = v___y_820_;
v___y_917_ = v___y_821_;
goto v___jp_914_;
}
else
{
size_t v___x_961_; size_t v___x_962_; lean_object* v___x_963_; 
v___x_961_ = ((size_t)0ULL);
v___x_962_ = lean_usize_of_nat(v___x_936_);
lean_inc_ref(v_rf_818_);
v___x_963_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0_spec__6(v_rf_818_, v_val_910_, v___x_961_, v___x_962_, v___x_959_, v___y_820_, v___y_821_);
if (lean_obj_tag(v___x_963_) == 0)
{
lean_object* v_a_964_; uint8_t v___x_965_; 
v_a_964_ = lean_ctor_get(v___x_963_, 0);
lean_inc(v_a_964_);
lean_dec_ref_known(v___x_963_, 1);
v___x_965_ = lean_unbox(v_a_964_);
lean_dec(v_a_964_);
v_anyErased_915_ = v___x_965_;
v___y_916_ = v___y_820_;
v___y_917_ = v___y_821_;
goto v___jp_914_;
}
else
{
lean_object* v_a_966_; lean_object* v___x_968_; uint8_t v_isShared_969_; uint8_t v_isSharedCheck_973_; 
lean_dec(v_val_910_);
lean_dec_ref(v_rf_818_);
v_a_966_ = lean_ctor_get(v___x_963_, 0);
v_isSharedCheck_973_ = !lean_is_exclusive(v___x_963_);
if (v_isSharedCheck_973_ == 0)
{
v___x_968_ = v___x_963_;
v_isShared_969_ = v_isSharedCheck_973_;
goto v_resetjp_967_;
}
else
{
lean_inc(v_a_966_);
lean_dec(v___x_963_);
v___x_968_ = lean_box(0);
v_isShared_969_ = v_isSharedCheck_973_;
goto v_resetjp_967_;
}
v_resetjp_967_:
{
lean_object* v___x_971_; 
if (v_isShared_969_ == 0)
{
v___x_971_ = v___x_968_;
goto v_reusejp_970_;
}
else
{
lean_object* v_reuseFailAlloc_972_; 
v_reuseFailAlloc_972_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_972_, 0, v_a_966_);
v___x_971_ = v_reuseFailAlloc_972_;
goto v_reusejp_970_;
}
v_reusejp_970_:
{
return v___x_971_;
}
}
}
}
}
else
{
size_t v___x_974_; size_t v___x_975_; lean_object* v___x_976_; 
v___x_974_ = ((size_t)0ULL);
v___x_975_ = lean_usize_of_nat(v___x_936_);
lean_inc_ref(v_rf_818_);
v___x_976_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0_spec__6(v_rf_818_, v_val_910_, v___x_974_, v___x_975_, v___x_959_, v___y_820_, v___y_821_);
if (lean_obj_tag(v___x_976_) == 0)
{
lean_object* v_a_977_; uint8_t v___x_978_; 
v_a_977_ = lean_ctor_get(v___x_976_, 0);
lean_inc(v_a_977_);
lean_dec_ref_known(v___x_976_, 1);
v___x_978_ = lean_unbox(v_a_977_);
lean_dec(v_a_977_);
v_anyErased_915_ = v___x_978_;
v___y_916_ = v___y_820_;
v___y_917_ = v___y_821_;
goto v___jp_914_;
}
else
{
lean_object* v_a_979_; lean_object* v___x_981_; uint8_t v_isShared_982_; uint8_t v_isSharedCheck_986_; 
lean_dec(v_val_910_);
lean_dec_ref(v_rf_818_);
v_a_979_ = lean_ctor_get(v___x_976_, 0);
v_isSharedCheck_986_ = !lean_is_exclusive(v___x_976_);
if (v_isSharedCheck_986_ == 0)
{
v___x_981_ = v___x_976_;
v_isShared_982_ = v_isSharedCheck_986_;
goto v_resetjp_980_;
}
else
{
lean_inc(v_a_979_);
lean_dec(v___x_976_);
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
}
v___jp_914_:
{
if (v_checkExists_819_ == 0)
{
lean_dec(v_val_910_);
lean_dec_ref(v_rf_818_);
goto v___jp_823_;
}
else
{
if (v_anyErased_915_ == 0)
{
lean_object* v_name_918_; lean_object* v___x_919_; lean_object* v___x_920_; lean_object* v___x_921_; lean_object* v___x_922_; lean_object* v___x_923_; size_t v_sz_924_; size_t v___x_925_; lean_object* v___x_926_; lean_object* v___x_927_; lean_object* v___x_928_; lean_object* v___x_929_; lean_object* v___x_930_; lean_object* v___x_931_; lean_object* v___x_932_; lean_object* v___x_933_; lean_object* v___x_934_; 
v_name_918_ = lean_ctor_get(v_rf_818_, 0);
lean_inc(v_name_918_);
lean_dec_ref(v_rf_818_);
v___x_919_ = lean_obj_once(&lp_aesop_Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0___closed__1, &lp_aesop_Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0___closed__1_once, _init_lp_aesop_Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0___closed__1);
v___x_920_ = l_Lean_MessageData_ofName(v_name_918_);
v___x_921_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_921_, 0, v___x_919_);
lean_ctor_set(v___x_921_, 1, v___x_920_);
v___x_922_ = lean_obj_once(&lp_aesop_Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0___closed__5, &lp_aesop_Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0___closed__5_once, _init_lp_aesop_Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0___closed__5);
v___x_923_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_923_, 0, v___x_921_);
lean_ctor_set(v___x_923_, 1, v___x_922_);
v_sz_924_ = lean_array_size(v_val_910_);
v___x_925_ = ((size_t)0ULL);
v___x_926_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0_spec__4(v_checkExists_819_, v_sz_924_, v___x_925_, v_val_910_);
v___x_927_ = lean_array_to_list(v___x_926_);
v___x_928_ = lean_box(0);
v___x_929_ = lp_aesop_List_mapTR_loop___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0_spec__5(v___x_927_, v___x_928_);
v___x_930_ = l_Lean_MessageData_ofList(v___x_929_);
v___x_931_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_931_, 0, v___x_923_);
lean_ctor_set(v___x_931_, 1, v___x_930_);
v___x_932_ = lean_obj_once(&lp_aesop_Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0___closed__7, &lp_aesop_Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0___closed__7_once, _init_lp_aesop_Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0___closed__7);
v___x_933_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_933_, 0, v___x_931_);
lean_ctor_set(v___x_933_, 1, v___x_932_);
v___x_934_ = lp_aesop_Lean_throwError___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0_spec__0___redArg(v___x_933_, v___y_916_, v___y_917_);
return v___x_934_;
}
else
{
lean_dec(v_val_910_);
lean_dec_ref(v_rf_818_);
goto v___jp_823_;
}
}
}
}
}
v___jp_823_:
{
lean_object* v___x_824_; lean_object* v___x_825_; 
v___x_824_ = lean_box(0);
v___x_825_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_825_, 0, v___x_824_);
return v___x_825_;
}
v___jp_826_:
{
lean_object* v___x_827_; lean_object* v___x_828_; 
v___x_827_ = lean_box(0);
v___x_828_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_828_, 0, v___x_827_);
return v___x_828_;
}
v___jp_829_:
{
if (v_checkExists_819_ == 0)
{
lean_dec_ref(v_rf_818_);
goto v___jp_826_;
}
else
{
if (v_anyErased_830_ == 0)
{
lean_object* v_name_833_; lean_object* v___x_834_; lean_object* v___x_835_; lean_object* v___x_836_; lean_object* v___x_837_; lean_object* v___x_838_; lean_object* v___x_839_; 
v_name_833_ = lean_ctor_get(v_rf_818_, 0);
lean_inc(v_name_833_);
lean_dec_ref(v_rf_818_);
v___x_834_ = lean_obj_once(&lp_aesop_Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0___closed__1, &lp_aesop_Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0___closed__1_once, _init_lp_aesop_Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0___closed__1);
v___x_835_ = l_Lean_MessageData_ofName(v_name_833_);
v___x_836_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_836_, 0, v___x_834_);
lean_ctor_set(v___x_836_, 1, v___x_835_);
v___x_837_ = lean_obj_once(&lp_aesop_Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0___closed__3, &lp_aesop_Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0___closed__3_once, _init_lp_aesop_Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0___closed__3);
v___x_838_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_838_, 0, v___x_836_);
lean_ctor_set(v___x_838_, 1, v___x_837_);
v___x_839_ = lp_aesop_Lean_throwError___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0_spec__0___redArg(v___x_838_, v___y_831_, v___y_832_);
return v___x_839_;
}
else
{
lean_dec_ref(v_rf_818_);
goto v___jp_826_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0___boxed(lean_object* v_rsf_988_, lean_object* v_rf_989_, lean_object* v_checkExists_990_, lean_object* v___y_991_, lean_object* v___y_992_, lean_object* v___y_993_){
_start:
{
uint8_t v_checkExists_boxed_994_; lean_object* v_res_995_; 
v_checkExists_boxed_994_ = lean_unbox(v_checkExists_990_);
v_res_995_ = lp_aesop_Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0(v_rsf_988_, v_rf_989_, v_checkExists_boxed_994_, v___y_991_, v___y_992_);
lean_dec(v___y_992_);
lean_dec_ref(v___y_991_);
return v_res_995_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___lam__1_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2_(lean_object* v_decl_998_, lean_object* v___y_999_, lean_object* v___y_1000_){
_start:
{
uint8_t v___x_1002_; lean_object* v___x_1003_; lean_object* v_ruleFilter_1004_; lean_object* v___x_1005_; uint8_t v___x_1006_; lean_object* v___x_1007_; 
v___x_1002_ = 0;
v___x_1003_ = ((lean_object*)(lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___lam__1___closed__0_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2_));
v_ruleFilter_1004_ = lean_alloc_ctor(0, 3, 1);
lean_ctor_set(v_ruleFilter_1004_, 0, v_decl_998_);
lean_ctor_set(v_ruleFilter_1004_, 1, v___x_1003_);
lean_ctor_set(v_ruleFilter_1004_, 2, v___x_1003_);
lean_ctor_set_uint8(v_ruleFilter_1004_, sizeof(void*)*3, v___x_1002_);
v___x_1005_ = lp_aesop_Aesop_RuleSetNameFilter_all;
v___x_1006_ = 1;
v___x_1007_ = lp_aesop_Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0(v___x_1005_, v_ruleFilter_1004_, v___x_1006_, v___y_999_, v___y_1000_);
return v___x_1007_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___lam__1_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2____boxed(lean_object* v_decl_1008_, lean_object* v___y_1009_, lean_object* v___y_1010_, lean_object* v___y_1011_){
_start:
{
lean_object* v_res_1012_; 
v_res_1012_ = lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___lam__1_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2_(v_decl_1008_, v___y_1009_, v___y_1010_);
lean_dec(v___y_1010_);
lean_dec_ref(v___y_1009_);
return v_res_1012_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__1(lean_object* v_decl_1013_, lean_object* v_as_1014_, size_t v_i_1015_, size_t v_stop_1016_, lean_object* v_b_1017_, lean_object* v___y_1018_, lean_object* v___y_1019_, lean_object* v___y_1020_, lean_object* v___y_1021_, lean_object* v___y_1022_, lean_object* v___y_1023_){
_start:
{
lean_object* v_a_1026_; uint8_t v___x_1030_; 
v___x_1030_ = lean_usize_dec_eq(v_i_1015_, v_stop_1016_);
if (v___x_1030_ == 0)
{
lean_object* v___x_1031_; lean_object* v___x_1032_; lean_object* v___x_1033_; 
v___x_1031_ = lean_array_uget_borrowed(v_as_1014_, v_i_1015_);
lean_inc(v_decl_1013_);
v___x_1032_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1032_, 0, v_decl_1013_);
lean_inc(v___x_1031_);
v___x_1033_ = lp_aesop_Aesop_Frontend_RuleExpr_buildAdditionalGlobalRules(v___x_1032_, v___x_1031_, v___y_1018_, v___y_1019_, v___y_1020_, v___y_1021_, v___y_1022_, v___y_1023_);
if (lean_obj_tag(v___x_1033_) == 0)
{
lean_object* v_a_1034_; lean_object* v___x_1035_; 
v_a_1034_ = lean_ctor_get(v___x_1033_, 0);
lean_inc(v_a_1034_);
lean_dec_ref_known(v___x_1033_, 1);
v___x_1035_ = l_Array_append___redArg(v_b_1017_, v_a_1034_);
lean_dec(v_a_1034_);
v_a_1026_ = v___x_1035_;
goto v___jp_1025_;
}
else
{
lean_dec_ref(v_b_1017_);
if (lean_obj_tag(v___x_1033_) == 0)
{
lean_object* v_a_1036_; 
v_a_1036_ = lean_ctor_get(v___x_1033_, 0);
lean_inc(v_a_1036_);
lean_dec_ref_known(v___x_1033_, 1);
v_a_1026_ = v_a_1036_;
goto v___jp_1025_;
}
else
{
lean_dec(v_decl_1013_);
return v___x_1033_;
}
}
}
else
{
lean_object* v___x_1037_; 
lean_dec(v_decl_1013_);
v___x_1037_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1037_, 0, v_b_1017_);
return v___x_1037_;
}
v___jp_1025_:
{
size_t v___x_1027_; size_t v___x_1028_; 
v___x_1027_ = ((size_t)1ULL);
v___x_1028_ = lean_usize_add(v_i_1015_, v___x_1027_);
v_i_1015_ = v___x_1028_;
v_b_1017_ = v_a_1026_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__1___boxed(lean_object* v_decl_1038_, lean_object* v_as_1039_, lean_object* v_i_1040_, lean_object* v_stop_1041_, lean_object* v_b_1042_, lean_object* v___y_1043_, lean_object* v___y_1044_, lean_object* v___y_1045_, lean_object* v___y_1046_, lean_object* v___y_1047_, lean_object* v___y_1048_, lean_object* v___y_1049_){
_start:
{
size_t v_i_boxed_1050_; size_t v_stop_boxed_1051_; lean_object* v_res_1052_; 
v_i_boxed_1050_ = lean_unbox_usize(v_i_1040_);
lean_dec(v_i_1040_);
v_stop_boxed_1051_ = lean_unbox_usize(v_stop_1041_);
lean_dec(v_stop_1041_);
v_res_1052_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__1(v_decl_1038_, v_as_1039_, v_i_boxed_1050_, v_stop_boxed_1051_, v_b_1042_, v___y_1043_, v___y_1044_, v___y_1045_, v___y_1046_, v___y_1047_, v___y_1048_);
lean_dec(v___y_1048_);
lean_dec_ref(v___y_1047_);
lean_dec(v___y_1046_);
lean_dec_ref(v___y_1045_);
lean_dec(v___y_1044_);
lean_dec_ref(v___y_1043_);
lean_dec_ref(v_as_1039_);
return v_res_1052_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___lam__2_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2_(lean_object* v_stx_1055_, lean_object* v_decl_1056_, lean_object* v___y_1057_, lean_object* v___y_1058_, lean_object* v___y_1059_, lean_object* v___y_1060_, lean_object* v___y_1061_, lean_object* v___y_1062_){
_start:
{
lean_object* v___x_1064_; 
v___x_1064_ = lp_aesop_Aesop_Frontend_AttrConfig_elab(v_stx_1055_, v___y_1057_, v___y_1058_, v___y_1059_, v___y_1060_, v___y_1061_, v___y_1062_);
if (lean_obj_tag(v___x_1064_) == 0)
{
lean_object* v_a_1065_; lean_object* v___x_1067_; uint8_t v_isShared_1068_; uint8_t v_isSharedCheck_1086_; 
v_a_1065_ = lean_ctor_get(v___x_1064_, 0);
v_isSharedCheck_1086_ = !lean_is_exclusive(v___x_1064_);
if (v_isSharedCheck_1086_ == 0)
{
v___x_1067_ = v___x_1064_;
v_isShared_1068_ = v_isSharedCheck_1086_;
goto v_resetjp_1066_;
}
else
{
lean_inc(v_a_1065_);
lean_dec(v___x_1064_);
v___x_1067_ = lean_box(0);
v_isShared_1068_ = v_isSharedCheck_1086_;
goto v_resetjp_1066_;
}
v_resetjp_1066_:
{
lean_object* v___x_1069_; lean_object* v___x_1070_; lean_object* v___x_1071_; uint8_t v___x_1072_; 
v___x_1069_ = lean_unsigned_to_nat(0u);
v___x_1070_ = ((lean_object*)(lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___lam__2___closed__0_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2_));
v___x_1071_ = lean_array_get_size(v_a_1065_);
v___x_1072_ = lean_nat_dec_lt(v___x_1069_, v___x_1071_);
if (v___x_1072_ == 0)
{
lean_object* v___x_1074_; 
lean_dec(v_a_1065_);
lean_dec(v_decl_1056_);
if (v_isShared_1068_ == 0)
{
lean_ctor_set(v___x_1067_, 0, v___x_1070_);
v___x_1074_ = v___x_1067_;
goto v_reusejp_1073_;
}
else
{
lean_object* v_reuseFailAlloc_1075_; 
v_reuseFailAlloc_1075_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1075_, 0, v___x_1070_);
v___x_1074_ = v_reuseFailAlloc_1075_;
goto v_reusejp_1073_;
}
v_reusejp_1073_:
{
return v___x_1074_;
}
}
else
{
uint8_t v___x_1076_; 
v___x_1076_ = lean_nat_dec_le(v___x_1071_, v___x_1071_);
if (v___x_1076_ == 0)
{
if (v___x_1072_ == 0)
{
lean_object* v___x_1078_; 
lean_dec(v_a_1065_);
lean_dec(v_decl_1056_);
if (v_isShared_1068_ == 0)
{
lean_ctor_set(v___x_1067_, 0, v___x_1070_);
v___x_1078_ = v___x_1067_;
goto v_reusejp_1077_;
}
else
{
lean_object* v_reuseFailAlloc_1079_; 
v_reuseFailAlloc_1079_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1079_, 0, v___x_1070_);
v___x_1078_ = v_reuseFailAlloc_1079_;
goto v_reusejp_1077_;
}
v_reusejp_1077_:
{
return v___x_1078_;
}
}
else
{
size_t v___x_1080_; size_t v___x_1081_; lean_object* v___x_1082_; 
lean_del_object(v___x_1067_);
v___x_1080_ = ((size_t)0ULL);
v___x_1081_ = lean_usize_of_nat(v___x_1071_);
v___x_1082_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__1(v_decl_1056_, v_a_1065_, v___x_1080_, v___x_1081_, v___x_1070_, v___y_1057_, v___y_1058_, v___y_1059_, v___y_1060_, v___y_1061_, v___y_1062_);
lean_dec(v_a_1065_);
return v___x_1082_;
}
}
else
{
size_t v___x_1083_; size_t v___x_1084_; lean_object* v___x_1085_; 
lean_del_object(v___x_1067_);
v___x_1083_ = ((size_t)0ULL);
v___x_1084_ = lean_usize_of_nat(v___x_1071_);
v___x_1085_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__1(v_decl_1056_, v_a_1065_, v___x_1083_, v___x_1084_, v___x_1070_, v___y_1057_, v___y_1058_, v___y_1059_, v___y_1060_, v___y_1061_, v___y_1062_);
lean_dec(v_a_1065_);
return v___x_1085_;
}
}
}
}
else
{
lean_object* v_a_1087_; lean_object* v___x_1089_; uint8_t v_isShared_1090_; uint8_t v_isSharedCheck_1094_; 
lean_dec(v_decl_1056_);
v_a_1087_ = lean_ctor_get(v___x_1064_, 0);
v_isSharedCheck_1094_ = !lean_is_exclusive(v___x_1064_);
if (v_isSharedCheck_1094_ == 0)
{
v___x_1089_ = v___x_1064_;
v_isShared_1090_ = v_isSharedCheck_1094_;
goto v_resetjp_1088_;
}
else
{
lean_inc(v_a_1087_);
lean_dec(v___x_1064_);
v___x_1089_ = lean_box(0);
v_isShared_1090_ = v_isSharedCheck_1094_;
goto v_resetjp_1088_;
}
v_resetjp_1088_:
{
lean_object* v___x_1092_; 
if (v_isShared_1090_ == 0)
{
v___x_1092_ = v___x_1089_;
goto v_reusejp_1091_;
}
else
{
lean_object* v_reuseFailAlloc_1093_; 
v_reuseFailAlloc_1093_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1093_, 0, v_a_1087_);
v___x_1092_ = v_reuseFailAlloc_1093_;
goto v_reusejp_1091_;
}
v_reusejp_1091_:
{
return v___x_1092_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___lam__2_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2____boxed(lean_object* v_stx_1095_, lean_object* v_decl_1096_, lean_object* v___y_1097_, lean_object* v___y_1098_, lean_object* v___y_1099_, lean_object* v___y_1100_, lean_object* v___y_1101_, lean_object* v___y_1102_, lean_object* v___y_1103_){
_start:
{
lean_object* v_res_1104_; 
v_res_1104_ = lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___lam__2_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2_(v_stx_1095_, v_decl_1096_, v___y_1097_, v___y_1098_, v___y_1099_, v___y_1100_, v___y_1101_, v___y_1102_);
lean_dec(v___y_1102_);
lean_dec_ref(v___y_1101_);
lean_dec(v___y_1100_);
lean_dec_ref(v___y_1099_);
lean_dec(v___y_1098_);
lean_dec_ref(v___y_1097_);
return v_res_1104_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_ScopedEnvExtension_add___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__10___redArg(lean_object* v_ext_1105_, lean_object* v_b_1106_, uint8_t v_kind_1107_, lean_object* v___y_1108_, lean_object* v___y_1109_){
_start:
{
lean_object* v_currNamespace_1111_; lean_object* v___x_1112_; lean_object* v_env_1113_; lean_object* v_nextMacroScope_1114_; lean_object* v_ngen_1115_; lean_object* v_auxDeclNGen_1116_; lean_object* v_traceState_1117_; lean_object* v_messages_1118_; lean_object* v_infoState_1119_; lean_object* v_snapshotTasks_1120_; lean_object* v___x_1122_; uint8_t v_isShared_1123_; uint8_t v_isSharedCheck_1132_; 
v_currNamespace_1111_ = lean_ctor_get(v___y_1108_, 6);
v___x_1112_ = lean_st_ref_take(v___y_1109_);
v_env_1113_ = lean_ctor_get(v___x_1112_, 0);
v_nextMacroScope_1114_ = lean_ctor_get(v___x_1112_, 1);
v_ngen_1115_ = lean_ctor_get(v___x_1112_, 2);
v_auxDeclNGen_1116_ = lean_ctor_get(v___x_1112_, 3);
v_traceState_1117_ = lean_ctor_get(v___x_1112_, 4);
v_messages_1118_ = lean_ctor_get(v___x_1112_, 6);
v_infoState_1119_ = lean_ctor_get(v___x_1112_, 7);
v_snapshotTasks_1120_ = lean_ctor_get(v___x_1112_, 8);
v_isSharedCheck_1132_ = !lean_is_exclusive(v___x_1112_);
if (v_isSharedCheck_1132_ == 0)
{
lean_object* v_unused_1133_; 
v_unused_1133_ = lean_ctor_get(v___x_1112_, 5);
lean_dec(v_unused_1133_);
v___x_1122_ = v___x_1112_;
v_isShared_1123_ = v_isSharedCheck_1132_;
goto v_resetjp_1121_;
}
else
{
lean_inc(v_snapshotTasks_1120_);
lean_inc(v_infoState_1119_);
lean_inc(v_messages_1118_);
lean_inc(v_traceState_1117_);
lean_inc(v_auxDeclNGen_1116_);
lean_inc(v_ngen_1115_);
lean_inc(v_nextMacroScope_1114_);
lean_inc(v_env_1113_);
lean_dec(v___x_1112_);
v___x_1122_ = lean_box(0);
v_isShared_1123_ = v_isSharedCheck_1132_;
goto v_resetjp_1121_;
}
v_resetjp_1121_:
{
lean_object* v___x_1124_; lean_object* v___x_1125_; lean_object* v___x_1127_; 
lean_inc(v_currNamespace_1111_);
v___x_1124_ = l_Lean_ScopedEnvExtension_addCore___redArg(v_env_1113_, v_ext_1105_, v_b_1106_, v_kind_1107_, v_currNamespace_1111_);
v___x_1125_ = lean_obj_once(&lp_aesop_Lean_setEnv___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__12___redArg___closed__2, &lp_aesop_Lean_setEnv___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__12___redArg___closed__2_once, _init_lp_aesop_Lean_setEnv___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__12___redArg___closed__2);
if (v_isShared_1123_ == 0)
{
lean_ctor_set(v___x_1122_, 5, v___x_1125_);
lean_ctor_set(v___x_1122_, 0, v___x_1124_);
v___x_1127_ = v___x_1122_;
goto v_reusejp_1126_;
}
else
{
lean_object* v_reuseFailAlloc_1131_; 
v_reuseFailAlloc_1131_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_1131_, 0, v___x_1124_);
lean_ctor_set(v_reuseFailAlloc_1131_, 1, v_nextMacroScope_1114_);
lean_ctor_set(v_reuseFailAlloc_1131_, 2, v_ngen_1115_);
lean_ctor_set(v_reuseFailAlloc_1131_, 3, v_auxDeclNGen_1116_);
lean_ctor_set(v_reuseFailAlloc_1131_, 4, v_traceState_1117_);
lean_ctor_set(v_reuseFailAlloc_1131_, 5, v___x_1125_);
lean_ctor_set(v_reuseFailAlloc_1131_, 6, v_messages_1118_);
lean_ctor_set(v_reuseFailAlloc_1131_, 7, v_infoState_1119_);
lean_ctor_set(v_reuseFailAlloc_1131_, 8, v_snapshotTasks_1120_);
v___x_1127_ = v_reuseFailAlloc_1131_;
goto v_reusejp_1126_;
}
v_reusejp_1126_:
{
lean_object* v___x_1128_; lean_object* v___x_1129_; lean_object* v___x_1130_; 
v___x_1128_ = lean_st_ref_set(v___y_1109_, v___x_1127_);
v___x_1129_ = lean_box(0);
v___x_1130_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1130_, 0, v___x_1129_);
return v___x_1130_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_ScopedEnvExtension_add___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__10___redArg___boxed(lean_object* v_ext_1134_, lean_object* v_b_1135_, lean_object* v_kind_1136_, lean_object* v___y_1137_, lean_object* v___y_1138_, lean_object* v___y_1139_){
_start:
{
uint8_t v_kind_boxed_1140_; lean_object* v_res_1141_; 
v_kind_boxed_1140_ = lean_unbox(v_kind_1136_);
v_res_1141_ = lp_aesop_Lean_ScopedEnvExtension_add___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__10___redArg(v_ext_1134_, v_b_1135_, v_kind_boxed_1140_, v___y_1137_, v___y_1138_);
lean_dec(v___y_1138_);
lean_dec_ref(v___y_1137_);
return v_res_1141_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Array_idxOfAux___at___00Array_finIdxOf_x3f___at___00Lean_PersistentHashMap_eraseAux___at___00Lean_PersistentHashMap_erase___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__11_spec__15_spec__21_spec__23(lean_object* v_xs_1142_, lean_object* v_v_1143_, lean_object* v_i_1144_){
_start:
{
uint8_t v___y_1150_; lean_object* v___x_1152_; uint8_t v___x_1153_; 
v___x_1152_ = lean_array_get_size(v_xs_1142_);
v___x_1153_ = lean_nat_dec_lt(v_i_1144_, v___x_1152_);
if (v___x_1153_ == 0)
{
lean_object* v___x_1154_; 
lean_dec(v_i_1144_);
v___x_1154_ = lean_box(0);
return v___x_1154_;
}
else
{
lean_object* v___x_1155_; 
v___x_1155_ = lean_array_fget_borrowed(v_xs_1142_, v_i_1144_);
if (lean_obj_tag(v___x_1155_) == 0)
{
if (lean_obj_tag(v_v_1143_) == 0)
{
lean_object* v_declName_1156_; uint8_t v_inv_1157_; lean_object* v_declName_1158_; uint8_t v_inv_1159_; uint8_t v___x_1160_; 
v_declName_1156_ = lean_ctor_get(v___x_1155_, 0);
v_inv_1157_ = lean_ctor_get_uint8(v___x_1155_, sizeof(void*)*1 + 1);
v_declName_1158_ = lean_ctor_get(v_v_1143_, 0);
v_inv_1159_ = lean_ctor_get_uint8(v_v_1143_, sizeof(void*)*1 + 1);
v___x_1160_ = lean_name_eq(v_declName_1156_, v_declName_1158_);
if (v___x_1160_ == 0)
{
v___y_1150_ = v___x_1160_;
goto v___jp_1149_;
}
else
{
if (v_inv_1157_ == 0)
{
if (v_inv_1159_ == 0)
{
v___y_1150_ = v___x_1160_;
goto v___jp_1149_;
}
else
{
goto v___jp_1145_;
}
}
else
{
v___y_1150_ = v_inv_1159_;
goto v___jp_1149_;
}
}
}
else
{
goto v___jp_1145_;
}
}
else
{
if (lean_obj_tag(v_v_1143_) == 0)
{
goto v___jp_1145_;
}
else
{
lean_object* v___x_1161_; lean_object* v___x_1162_; uint8_t v___x_1163_; 
v___x_1161_ = l_Lean_Meta_Origin_key(v___x_1155_);
v___x_1162_ = l_Lean_Meta_Origin_key(v_v_1143_);
v___x_1163_ = lean_name_eq(v___x_1161_, v___x_1162_);
lean_dec(v___x_1162_);
lean_dec(v___x_1161_);
v___y_1150_ = v___x_1163_;
goto v___jp_1149_;
}
}
}
v___jp_1145_:
{
lean_object* v___x_1146_; lean_object* v___x_1147_; 
v___x_1146_ = lean_unsigned_to_nat(1u);
v___x_1147_ = lean_nat_add(v_i_1144_, v___x_1146_);
lean_dec(v_i_1144_);
v_i_1144_ = v___x_1147_;
goto _start;
}
v___jp_1149_:
{
if (v___y_1150_ == 0)
{
goto v___jp_1145_;
}
else
{
lean_object* v___x_1151_; 
v___x_1151_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1151_, 0, v_i_1144_);
return v___x_1151_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Array_idxOfAux___at___00Array_finIdxOf_x3f___at___00Lean_PersistentHashMap_eraseAux___at___00Lean_PersistentHashMap_erase___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__11_spec__15_spec__21_spec__23___boxed(lean_object* v_xs_1164_, lean_object* v_v_1165_, lean_object* v_i_1166_){
_start:
{
lean_object* v_res_1167_; 
v_res_1167_ = lp_aesop_Array_idxOfAux___at___00Array_finIdxOf_x3f___at___00Lean_PersistentHashMap_eraseAux___at___00Lean_PersistentHashMap_erase___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__11_spec__15_spec__21_spec__23(v_xs_1164_, v_v_1165_, v_i_1166_);
lean_dec_ref(v_v_1165_);
lean_dec_ref(v_xs_1164_);
return v_res_1167_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Array_finIdxOf_x3f___at___00Lean_PersistentHashMap_eraseAux___at___00Lean_PersistentHashMap_erase___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__11_spec__15_spec__21(lean_object* v_xs_1168_, lean_object* v_v_1169_){
_start:
{
lean_object* v___x_1170_; lean_object* v___x_1171_; 
v___x_1170_ = lean_unsigned_to_nat(0u);
v___x_1171_ = lp_aesop_Array_idxOfAux___at___00Array_finIdxOf_x3f___at___00Lean_PersistentHashMap_eraseAux___at___00Lean_PersistentHashMap_erase___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__11_spec__15_spec__21_spec__23(v_xs_1168_, v_v_1169_, v___x_1170_);
return v___x_1171_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Array_finIdxOf_x3f___at___00Lean_PersistentHashMap_eraseAux___at___00Lean_PersistentHashMap_erase___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__11_spec__15_spec__21___boxed(lean_object* v_xs_1172_, lean_object* v_v_1173_){
_start:
{
lean_object* v_res_1174_; 
v_res_1174_ = lp_aesop_Array_finIdxOf_x3f___at___00Lean_PersistentHashMap_eraseAux___at___00Lean_PersistentHashMap_erase___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__11_spec__15_spec__21(v_xs_1172_, v_v_1173_);
lean_dec_ref(v_v_1173_);
lean_dec_ref(v_xs_1172_);
return v_res_1174_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_eraseAux___at___00Lean_PersistentHashMap_erase___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__11_spec__15___redArg(lean_object* v_x_1175_, size_t v_x_1176_, lean_object* v_x_1177_){
_start:
{
if (lean_obj_tag(v_x_1175_) == 0)
{
lean_object* v_es_1178_; lean_object* v___x_1179_; size_t v___x_1180_; size_t v___x_1181_; lean_object* v_j_1182_; uint8_t v___y_1184_; lean_object* v_entry_1194_; 
v_es_1178_ = lean_ctor_get(v_x_1175_, 0);
v___x_1179_ = lean_box(2);
v___x_1180_ = ((size_t)31ULL);
v___x_1181_ = lean_usize_land(v_x_1176_, v___x_1180_);
v_j_1182_ = lean_usize_to_nat(v___x_1181_);
v_entry_1194_ = lean_array_get(v___x_1179_, v_es_1178_, v_j_1182_);
switch(lean_obj_tag(v_entry_1194_))
{
case 0:
{
if (lean_obj_tag(v_x_1177_) == 0)
{
lean_object* v_key_1195_; 
v_key_1195_ = lean_ctor_get(v_entry_1194_, 0);
lean_inc(v_key_1195_);
lean_dec_ref_known(v_entry_1194_, 2);
if (lean_obj_tag(v_key_1195_) == 0)
{
lean_object* v_declName_1196_; uint8_t v_inv_1197_; lean_object* v_declName_1198_; uint8_t v_inv_1199_; uint8_t v___x_1200_; 
v_declName_1196_ = lean_ctor_get(v_x_1177_, 0);
v_inv_1197_ = lean_ctor_get_uint8(v_x_1177_, sizeof(void*)*1 + 1);
v_declName_1198_ = lean_ctor_get(v_key_1195_, 0);
lean_inc(v_declName_1198_);
v_inv_1199_ = lean_ctor_get_uint8(v_key_1195_, sizeof(void*)*1 + 1);
lean_dec_ref_known(v_key_1195_, 1);
v___x_1200_ = lean_name_eq(v_declName_1196_, v_declName_1198_);
lean_dec(v_declName_1198_);
if (v___x_1200_ == 0)
{
v___y_1184_ = v___x_1200_;
goto v___jp_1183_;
}
else
{
if (v_inv_1197_ == 0)
{
if (v_inv_1199_ == 0)
{
v___y_1184_ = v___x_1200_;
goto v___jp_1183_;
}
else
{
lean_dec(v_j_1182_);
return v_x_1175_;
}
}
else
{
v___y_1184_ = v_inv_1199_;
goto v___jp_1183_;
}
}
}
else
{
lean_dec(v_key_1195_);
lean_dec(v_j_1182_);
return v_x_1175_;
}
}
else
{
lean_object* v_key_1201_; 
v_key_1201_ = lean_ctor_get(v_entry_1194_, 0);
lean_inc(v_key_1201_);
lean_dec_ref_known(v_entry_1194_, 2);
if (lean_obj_tag(v_key_1201_) == 0)
{
lean_dec_ref_known(v_key_1201_, 1);
lean_dec(v_j_1182_);
return v_x_1175_;
}
else
{
lean_object* v___x_1202_; lean_object* v___x_1203_; uint8_t v___x_1204_; 
v___x_1202_ = l_Lean_Meta_Origin_key(v_x_1177_);
v___x_1203_ = l_Lean_Meta_Origin_key(v_key_1201_);
lean_dec(v_key_1201_);
v___x_1204_ = lean_name_eq(v___x_1202_, v___x_1203_);
lean_dec(v___x_1203_);
lean_dec(v___x_1202_);
v___y_1184_ = v___x_1204_;
goto v___jp_1183_;
}
}
}
case 1:
{
lean_object* v___x_1206_; uint8_t v_isShared_1207_; uint8_t v_isSharedCheck_1239_; 
lean_inc_ref(v_es_1178_);
v_isSharedCheck_1239_ = !lean_is_exclusive(v_x_1175_);
if (v_isSharedCheck_1239_ == 0)
{
lean_object* v_unused_1240_; 
v_unused_1240_ = lean_ctor_get(v_x_1175_, 0);
lean_dec(v_unused_1240_);
v___x_1206_ = v_x_1175_;
v_isShared_1207_ = v_isSharedCheck_1239_;
goto v_resetjp_1205_;
}
else
{
lean_dec(v_x_1175_);
v___x_1206_ = lean_box(0);
v_isShared_1207_ = v_isSharedCheck_1239_;
goto v_resetjp_1205_;
}
v_resetjp_1205_:
{
lean_object* v_node_1208_; lean_object* v___x_1210_; uint8_t v_isShared_1211_; uint8_t v_isSharedCheck_1238_; 
v_node_1208_ = lean_ctor_get(v_entry_1194_, 0);
v_isSharedCheck_1238_ = !lean_is_exclusive(v_entry_1194_);
if (v_isSharedCheck_1238_ == 0)
{
v___x_1210_ = v_entry_1194_;
v_isShared_1211_ = v_isSharedCheck_1238_;
goto v_resetjp_1209_;
}
else
{
lean_inc(v_node_1208_);
lean_dec(v_entry_1194_);
v___x_1210_ = lean_box(0);
v_isShared_1211_ = v_isSharedCheck_1238_;
goto v_resetjp_1209_;
}
v_resetjp_1209_:
{
size_t v___x_1212_; lean_object* v_entries_1213_; size_t v___x_1214_; lean_object* v_newNode_1215_; lean_object* v___x_1216_; 
v___x_1212_ = ((size_t)5ULL);
v_entries_1213_ = lean_array_set(v_es_1178_, v_j_1182_, v___x_1179_);
v___x_1214_ = lean_usize_shift_right(v_x_1176_, v___x_1212_);
v_newNode_1215_ = lp_aesop_Lean_PersistentHashMap_eraseAux___at___00Lean_PersistentHashMap_erase___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__11_spec__15___redArg(v_node_1208_, v___x_1214_, v_x_1177_);
lean_inc_ref(v_newNode_1215_);
v___x_1216_ = l_Lean_PersistentHashMap_isUnaryNode___redArg(v_newNode_1215_);
if (lean_obj_tag(v___x_1216_) == 0)
{
lean_object* v___x_1218_; 
if (v_isShared_1211_ == 0)
{
lean_ctor_set(v___x_1210_, 0, v_newNode_1215_);
v___x_1218_ = v___x_1210_;
goto v_reusejp_1217_;
}
else
{
lean_object* v_reuseFailAlloc_1223_; 
v_reuseFailAlloc_1223_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1223_, 0, v_newNode_1215_);
v___x_1218_ = v_reuseFailAlloc_1223_;
goto v_reusejp_1217_;
}
v_reusejp_1217_:
{
lean_object* v___x_1219_; lean_object* v___x_1221_; 
v___x_1219_ = lean_array_set(v_entries_1213_, v_j_1182_, v___x_1218_);
lean_dec(v_j_1182_);
if (v_isShared_1207_ == 0)
{
lean_ctor_set(v___x_1206_, 0, v___x_1219_);
v___x_1221_ = v___x_1206_;
goto v_reusejp_1220_;
}
else
{
lean_object* v_reuseFailAlloc_1222_; 
v_reuseFailAlloc_1222_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1222_, 0, v___x_1219_);
v___x_1221_ = v_reuseFailAlloc_1222_;
goto v_reusejp_1220_;
}
v_reusejp_1220_:
{
return v___x_1221_;
}
}
}
else
{
lean_object* v_val_1224_; lean_object* v_fst_1225_; lean_object* v_snd_1226_; lean_object* v___x_1228_; uint8_t v_isShared_1229_; uint8_t v_isSharedCheck_1237_; 
lean_dec_ref(v_newNode_1215_);
lean_del_object(v___x_1210_);
v_val_1224_ = lean_ctor_get(v___x_1216_, 0);
lean_inc(v_val_1224_);
lean_dec_ref_known(v___x_1216_, 1);
v_fst_1225_ = lean_ctor_get(v_val_1224_, 0);
v_snd_1226_ = lean_ctor_get(v_val_1224_, 1);
v_isSharedCheck_1237_ = !lean_is_exclusive(v_val_1224_);
if (v_isSharedCheck_1237_ == 0)
{
v___x_1228_ = v_val_1224_;
v_isShared_1229_ = v_isSharedCheck_1237_;
goto v_resetjp_1227_;
}
else
{
lean_inc(v_snd_1226_);
lean_inc(v_fst_1225_);
lean_dec(v_val_1224_);
v___x_1228_ = lean_box(0);
v_isShared_1229_ = v_isSharedCheck_1237_;
goto v_resetjp_1227_;
}
v_resetjp_1227_:
{
lean_object* v___x_1231_; 
if (v_isShared_1229_ == 0)
{
v___x_1231_ = v___x_1228_;
goto v_reusejp_1230_;
}
else
{
lean_object* v_reuseFailAlloc_1236_; 
v_reuseFailAlloc_1236_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1236_, 0, v_fst_1225_);
lean_ctor_set(v_reuseFailAlloc_1236_, 1, v_snd_1226_);
v___x_1231_ = v_reuseFailAlloc_1236_;
goto v_reusejp_1230_;
}
v_reusejp_1230_:
{
lean_object* v___x_1232_; lean_object* v___x_1234_; 
v___x_1232_ = lean_array_set(v_entries_1213_, v_j_1182_, v___x_1231_);
lean_dec(v_j_1182_);
if (v_isShared_1207_ == 0)
{
lean_ctor_set(v___x_1206_, 0, v___x_1232_);
v___x_1234_ = v___x_1206_;
goto v_reusejp_1233_;
}
else
{
lean_object* v_reuseFailAlloc_1235_; 
v_reuseFailAlloc_1235_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1235_, 0, v___x_1232_);
v___x_1234_ = v_reuseFailAlloc_1235_;
goto v_reusejp_1233_;
}
v_reusejp_1233_:
{
return v___x_1234_;
}
}
}
}
}
}
}
default: 
{
lean_dec(v_j_1182_);
return v_x_1175_;
}
}
v___jp_1183_:
{
if (v___y_1184_ == 0)
{
lean_dec(v_j_1182_);
return v_x_1175_;
}
else
{
lean_object* v___x_1186_; uint8_t v_isShared_1187_; uint8_t v_isSharedCheck_1192_; 
lean_inc_ref(v_es_1178_);
v_isSharedCheck_1192_ = !lean_is_exclusive(v_x_1175_);
if (v_isSharedCheck_1192_ == 0)
{
lean_object* v_unused_1193_; 
v_unused_1193_ = lean_ctor_get(v_x_1175_, 0);
lean_dec(v_unused_1193_);
v___x_1186_ = v_x_1175_;
v_isShared_1187_ = v_isSharedCheck_1192_;
goto v_resetjp_1185_;
}
else
{
lean_dec(v_x_1175_);
v___x_1186_ = lean_box(0);
v_isShared_1187_ = v_isSharedCheck_1192_;
goto v_resetjp_1185_;
}
v_resetjp_1185_:
{
lean_object* v___x_1188_; lean_object* v___x_1190_; 
v___x_1188_ = lean_array_set(v_es_1178_, v_j_1182_, v___x_1179_);
lean_dec(v_j_1182_);
if (v_isShared_1187_ == 0)
{
lean_ctor_set(v___x_1186_, 0, v___x_1188_);
v___x_1190_ = v___x_1186_;
goto v_reusejp_1189_;
}
else
{
lean_object* v_reuseFailAlloc_1191_; 
v_reuseFailAlloc_1191_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1191_, 0, v___x_1188_);
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
else
{
lean_object* v_ks_1241_; lean_object* v_vs_1242_; lean_object* v___x_1244_; uint8_t v_isShared_1245_; uint8_t v_isSharedCheck_1256_; 
v_ks_1241_ = lean_ctor_get(v_x_1175_, 0);
v_vs_1242_ = lean_ctor_get(v_x_1175_, 1);
v_isSharedCheck_1256_ = !lean_is_exclusive(v_x_1175_);
if (v_isSharedCheck_1256_ == 0)
{
v___x_1244_ = v_x_1175_;
v_isShared_1245_ = v_isSharedCheck_1256_;
goto v_resetjp_1243_;
}
else
{
lean_inc(v_vs_1242_);
lean_inc(v_ks_1241_);
lean_dec(v_x_1175_);
v___x_1244_ = lean_box(0);
v_isShared_1245_ = v_isSharedCheck_1256_;
goto v_resetjp_1243_;
}
v_resetjp_1243_:
{
lean_object* v___x_1246_; 
v___x_1246_ = lp_aesop_Array_finIdxOf_x3f___at___00Lean_PersistentHashMap_eraseAux___at___00Lean_PersistentHashMap_erase___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__11_spec__15_spec__21(v_ks_1241_, v_x_1177_);
if (lean_obj_tag(v___x_1246_) == 0)
{
lean_object* v___x_1248_; 
if (v_isShared_1245_ == 0)
{
v___x_1248_ = v___x_1244_;
goto v_reusejp_1247_;
}
else
{
lean_object* v_reuseFailAlloc_1249_; 
v_reuseFailAlloc_1249_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1249_, 0, v_ks_1241_);
lean_ctor_set(v_reuseFailAlloc_1249_, 1, v_vs_1242_);
v___x_1248_ = v_reuseFailAlloc_1249_;
goto v_reusejp_1247_;
}
v_reusejp_1247_:
{
return v___x_1248_;
}
}
else
{
lean_object* v_val_1250_; lean_object* v_keys_x27_1251_; lean_object* v_vals_x27_1252_; lean_object* v___x_1254_; 
v_val_1250_ = lean_ctor_get(v___x_1246_, 0);
lean_inc_n(v_val_1250_, 2);
lean_dec_ref_known(v___x_1246_, 1);
v_keys_x27_1251_ = l_Array_eraseIdx___redArg(v_ks_1241_, v_val_1250_);
v_vals_x27_1252_ = l_Array_eraseIdx___redArg(v_vs_1242_, v_val_1250_);
if (v_isShared_1245_ == 0)
{
lean_ctor_set(v___x_1244_, 1, v_vals_x27_1252_);
lean_ctor_set(v___x_1244_, 0, v_keys_x27_1251_);
v___x_1254_ = v___x_1244_;
goto v_reusejp_1253_;
}
else
{
lean_object* v_reuseFailAlloc_1255_; 
v_reuseFailAlloc_1255_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1255_, 0, v_keys_x27_1251_);
lean_ctor_set(v_reuseFailAlloc_1255_, 1, v_vals_x27_1252_);
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
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_eraseAux___at___00Lean_PersistentHashMap_erase___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__11_spec__15___redArg___boxed(lean_object* v_x_1257_, lean_object* v_x_1258_, lean_object* v_x_1259_){
_start:
{
size_t v_x_10755__boxed_1260_; lean_object* v_res_1261_; 
v_x_10755__boxed_1260_ = lean_unbox_usize(v_x_1258_);
lean_dec(v_x_1258_);
v_res_1261_ = lp_aesop_Lean_PersistentHashMap_eraseAux___at___00Lean_PersistentHashMap_erase___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__11_spec__15___redArg(v_x_1257_, v_x_10755__boxed_1260_, v_x_1259_);
lean_dec_ref(v_x_1259_);
return v_res_1261_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_erase___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__11___redArg(lean_object* v_x_1262_, lean_object* v_x_1263_){
_start:
{
uint64_t v___y_1265_; uint64_t v___y_1269_; uint64_t v___y_1273_; 
if (lean_obj_tag(v_x_1263_) == 0)
{
uint8_t v_inv_1276_; 
v_inv_1276_ = lean_ctor_get_uint8(v_x_1263_, sizeof(void*)*1 + 1);
if (v_inv_1276_ == 0)
{
lean_object* v_declName_1277_; 
v_declName_1277_ = lean_ctor_get(v_x_1263_, 0);
if (lean_obj_tag(v_declName_1277_) == 0)
{
uint64_t v___x_1278_; 
v___x_1278_ = 1723ULL;
v___y_1269_ = v___x_1278_;
goto v___jp_1268_;
}
else
{
uint64_t v_hash_1279_; 
v_hash_1279_ = lean_ctor_get_uint64(v_declName_1277_, sizeof(void*)*2);
v___y_1269_ = v_hash_1279_;
goto v___jp_1268_;
}
}
else
{
lean_object* v_declName_1280_; 
v_declName_1280_ = lean_ctor_get(v_x_1263_, 0);
if (lean_obj_tag(v_declName_1280_) == 0)
{
uint64_t v___x_1281_; 
v___x_1281_ = 1723ULL;
v___y_1273_ = v___x_1281_;
goto v___jp_1272_;
}
else
{
uint64_t v_hash_1282_; 
v_hash_1282_ = lean_ctor_get_uint64(v_declName_1280_, sizeof(void*)*2);
v___y_1273_ = v_hash_1282_;
goto v___jp_1272_;
}
}
}
else
{
lean_object* v___x_1283_; 
v___x_1283_ = l_Lean_Meta_Origin_key(v_x_1263_);
if (lean_obj_tag(v___x_1283_) == 0)
{
uint64_t v___x_1284_; 
v___x_1284_ = 1723ULL;
v___y_1265_ = v___x_1284_;
goto v___jp_1264_;
}
else
{
uint64_t v_hash_1285_; 
v_hash_1285_ = lean_ctor_get_uint64(v___x_1283_, sizeof(void*)*2);
lean_dec(v___x_1283_);
v___y_1265_ = v_hash_1285_;
goto v___jp_1264_;
}
}
v___jp_1264_:
{
size_t v_h_1266_; lean_object* v___x_1267_; 
v_h_1266_ = lean_uint64_to_usize(v___y_1265_);
v___x_1267_ = lp_aesop_Lean_PersistentHashMap_eraseAux___at___00Lean_PersistentHashMap_erase___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__11_spec__15___redArg(v_x_1262_, v_h_1266_, v_x_1263_);
return v___x_1267_;
}
v___jp_1268_:
{
uint64_t v___x_1270_; uint64_t v___x_1271_; 
v___x_1270_ = 13ULL;
v___x_1271_ = lean_uint64_mix_hash(v___y_1269_, v___x_1270_);
v___y_1265_ = v___x_1271_;
goto v___jp_1264_;
}
v___jp_1272_:
{
uint64_t v___x_1274_; uint64_t v___x_1275_; 
v___x_1274_ = 11ULL;
v___x_1275_ = lean_uint64_mix_hash(v___y_1273_, v___x_1274_);
v___y_1265_ = v___x_1275_;
goto v___jp_1264_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_erase___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__11___redArg___boxed(lean_object* v_x_1286_, lean_object* v_x_1287_){
_start:
{
lean_object* v_res_1288_; 
v_res_1288_ = lp_aesop_Lean_PersistentHashMap_erase___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__11___redArg(v_x_1286_, v_x_1287_);
lean_dec_ref(v_x_1287_);
return v_res_1288_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__13___lam__0(lean_object* v_a_1289_, lean_object* v_simpTheorems_1290_){
_start:
{
lean_object* v_pre_1291_; lean_object* v_post_1292_; lean_object* v_lemmaNames_1293_; lean_object* v_toUnfold_1294_; lean_object* v_erased_1295_; lean_object* v_toUnfoldThms_1296_; lean_object* v___x_1298_; uint8_t v_isShared_1299_; uint8_t v_isSharedCheck_1305_; 
v_pre_1291_ = lean_ctor_get(v_simpTheorems_1290_, 0);
v_post_1292_ = lean_ctor_get(v_simpTheorems_1290_, 1);
v_lemmaNames_1293_ = lean_ctor_get(v_simpTheorems_1290_, 2);
v_toUnfold_1294_ = lean_ctor_get(v_simpTheorems_1290_, 3);
v_erased_1295_ = lean_ctor_get(v_simpTheorems_1290_, 4);
v_toUnfoldThms_1296_ = lean_ctor_get(v_simpTheorems_1290_, 5);
v_isSharedCheck_1305_ = !lean_is_exclusive(v_simpTheorems_1290_);
if (v_isSharedCheck_1305_ == 0)
{
v___x_1298_ = v_simpTheorems_1290_;
v_isShared_1299_ = v_isSharedCheck_1305_;
goto v_resetjp_1297_;
}
else
{
lean_inc(v_toUnfoldThms_1296_);
lean_inc(v_erased_1295_);
lean_inc(v_toUnfold_1294_);
lean_inc(v_lemmaNames_1293_);
lean_inc(v_post_1292_);
lean_inc(v_pre_1291_);
lean_dec(v_simpTheorems_1290_);
v___x_1298_ = lean_box(0);
v_isShared_1299_ = v_isSharedCheck_1305_;
goto v_resetjp_1297_;
}
v_resetjp_1297_:
{
lean_object* v_origin_1300_; lean_object* v___x_1301_; lean_object* v___x_1303_; 
v_origin_1300_ = lean_ctor_get(v_a_1289_, 4);
v___x_1301_ = lp_aesop_Lean_PersistentHashMap_erase___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__11___redArg(v_erased_1295_, v_origin_1300_);
if (v_isShared_1299_ == 0)
{
lean_ctor_set(v___x_1298_, 4, v___x_1301_);
v___x_1303_ = v___x_1298_;
goto v_reusejp_1302_;
}
else
{
lean_object* v_reuseFailAlloc_1304_; 
v_reuseFailAlloc_1304_ = lean_alloc_ctor(0, 6, 0);
lean_ctor_set(v_reuseFailAlloc_1304_, 0, v_pre_1291_);
lean_ctor_set(v_reuseFailAlloc_1304_, 1, v_post_1292_);
lean_ctor_set(v_reuseFailAlloc_1304_, 2, v_lemmaNames_1293_);
lean_ctor_set(v_reuseFailAlloc_1304_, 3, v_toUnfold_1294_);
lean_ctor_set(v_reuseFailAlloc_1304_, 4, v___x_1301_);
lean_ctor_set(v_reuseFailAlloc_1304_, 5, v_toUnfoldThms_1296_);
v___x_1303_ = v_reuseFailAlloc_1304_;
goto v_reusejp_1302_;
}
v_reusejp_1302_:
{
return v___x_1303_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__13___lam__0___boxed(lean_object* v_a_1306_, lean_object* v_simpTheorems_1307_){
_start:
{
lean_object* v_res_1308_; 
v_res_1308_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__13___lam__0(v_a_1306_, v_simpTheorems_1307_);
lean_dec_ref(v_a_1306_);
return v_res_1308_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__13(lean_object* v_fst_1309_, uint8_t v_kind_1310_, lean_object* v_as_1311_, size_t v_sz_1312_, size_t v_i_1313_, lean_object* v_b_1314_, lean_object* v___y_1315_, lean_object* v___y_1316_){
_start:
{
lean_object* v_a_1319_; uint8_t v___x_1323_; 
v___x_1323_ = lean_usize_dec_lt(v_i_1313_, v_sz_1312_);
if (v___x_1323_ == 0)
{
lean_object* v___x_1324_; 
lean_dec_ref(v_fst_1309_);
v___x_1324_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1324_, 0, v_b_1314_);
return v___x_1324_;
}
else
{
lean_object* v_a_1325_; lean_object* v___x_1326_; 
v_a_1325_ = lean_array_uget_borrowed(v_as_1311_, v_i_1313_);
lean_inc(v_a_1325_);
lean_inc_ref(v_fst_1309_);
v___x_1326_ = lp_aesop_Lean_ScopedEnvExtension_add___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__10___redArg(v_fst_1309_, v_a_1325_, v_kind_1310_, v___y_1315_, v___y_1316_);
if (lean_obj_tag(v___x_1326_) == 0)
{
lean_object* v___x_1327_; 
lean_dec_ref_known(v___x_1326_, 1);
v___x_1327_ = lean_box(0);
if (lean_obj_tag(v_a_1325_) == 0)
{
lean_object* v_a_1328_; lean_object* v___x_1329_; lean_object* v_env_1330_; lean_object* v___f_1331_; lean_object* v___x_1332_; lean_object* v___x_1333_; 
v_a_1328_ = lean_ctor_get(v_a_1325_, 0);
v___x_1329_ = lean_st_ref_get(v___y_1316_);
v_env_1330_ = lean_ctor_get(v___x_1329_, 0);
lean_inc_ref(v_env_1330_);
lean_dec(v___x_1329_);
lean_inc_ref(v_a_1328_);
v___f_1331_ = lean_alloc_closure((void*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__13___lam__0___boxed), 2, 1);
lean_closure_set(v___f_1331_, 0, v_a_1328_);
lean_inc_ref(v_fst_1309_);
v___x_1332_ = l_Lean_ScopedEnvExtension_modifyState___redArg(v_fst_1309_, v_env_1330_, v___f_1331_);
v___x_1333_ = lp_aesop_Lean_setEnv___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__12___redArg(v___x_1332_, v___y_1316_);
if (lean_obj_tag(v___x_1333_) == 0)
{
lean_dec_ref_known(v___x_1333_, 1);
v_a_1319_ = v___x_1327_;
goto v___jp_1318_;
}
else
{
lean_dec_ref(v_fst_1309_);
return v___x_1333_;
}
}
else
{
v_a_1319_ = v___x_1327_;
goto v___jp_1318_;
}
}
else
{
lean_dec_ref(v_fst_1309_);
return v___x_1326_;
}
}
v___jp_1318_:
{
size_t v___x_1320_; size_t v___x_1321_; 
v___x_1320_ = ((size_t)1ULL);
v___x_1321_ = lean_usize_add(v_i_1313_, v___x_1320_);
v_i_1313_ = v___x_1321_;
v_b_1314_ = v_a_1319_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__13___boxed(lean_object* v_fst_1334_, lean_object* v_kind_1335_, lean_object* v_as_1336_, lean_object* v_sz_1337_, lean_object* v_i_1338_, lean_object* v_b_1339_, lean_object* v___y_1340_, lean_object* v___y_1341_, lean_object* v___y_1342_){
_start:
{
uint8_t v_kind_boxed_1343_; size_t v_sz_boxed_1344_; size_t v_i_boxed_1345_; lean_object* v_res_1346_; 
v_kind_boxed_1343_ = lean_unbox(v_kind_1335_);
v_sz_boxed_1344_ = lean_unbox_usize(v_sz_1337_);
lean_dec(v_sz_1337_);
v_i_boxed_1345_ = lean_unbox_usize(v_i_1338_);
lean_dec(v_i_1338_);
v_res_1346_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__13(v_fst_1334_, v_kind_boxed_1343_, v_as_1336_, v_sz_boxed_1344_, v_i_boxed_1345_, v_b_1339_, v___y_1340_, v___y_1341_);
lean_dec(v___y_1341_);
lean_dec_ref(v___y_1340_);
lean_dec_ref(v_as_1336_);
return v_res_1346_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_getGlobalRuleSetFromData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__14___redArg(lean_object* v_ext_1347_, lean_object* v_simpExt_1348_, lean_object* v_simprocExt_1349_, lean_object* v___y_1350_){
_start:
{
lean_object* v___x_1352_; lean_object* v_ext_1353_; lean_object* v_toEnvExtension_1354_; lean_object* v_ext_1355_; lean_object* v_toEnvExtension_1356_; lean_object* v_ext_1357_; lean_object* v_toEnvExtension_1358_; lean_object* v_env_1359_; lean_object* v_asyncMode_1360_; lean_object* v_asyncMode_1361_; lean_object* v_asyncMode_1362_; lean_object* v___x_1363_; lean_object* v_base_1364_; lean_object* v___x_1365_; lean_object* v___x_1366_; lean_object* v_simpTheorems_1367_; lean_object* v_simprocs_1368_; lean_object* v___x_1369_; lean_object* v___x_1370_; 
v___x_1352_ = lean_st_ref_get(v___y_1350_);
v_ext_1353_ = lean_ctor_get(v_ext_1347_, 1);
v_toEnvExtension_1354_ = lean_ctor_get(v_ext_1353_, 0);
v_ext_1355_ = lean_ctor_get(v_simpExt_1348_, 1);
v_toEnvExtension_1356_ = lean_ctor_get(v_ext_1355_, 0);
v_ext_1357_ = lean_ctor_get(v_simprocExt_1349_, 1);
v_toEnvExtension_1358_ = lean_ctor_get(v_ext_1357_, 0);
v_env_1359_ = lean_ctor_get(v___x_1352_, 0);
lean_inc_ref_n(v_env_1359_, 3);
lean_dec(v___x_1352_);
v_asyncMode_1360_ = lean_ctor_get(v_toEnvExtension_1354_, 2);
v_asyncMode_1361_ = lean_ctor_get(v_toEnvExtension_1356_, 2);
v_asyncMode_1362_ = lean_ctor_get(v_toEnvExtension_1358_, 2);
v___x_1363_ = lp_aesop_Aesop_instInhabitedBaseRuleSet_default;
v_base_1364_ = l_Lean_ScopedEnvExtension_getState___redArg(v___x_1363_, v_ext_1347_, v_env_1359_, v_asyncMode_1360_);
v___x_1365_ = l_Lean_Meta_instInhabitedSimpTheorems_default;
v___x_1366_ = l_Lean_Meta_Simp_instInhabitedSimprocs_default;
v_simpTheorems_1367_ = l_Lean_ScopedEnvExtension_getState___redArg(v___x_1365_, v_simpExt_1348_, v_env_1359_, v_asyncMode_1361_);
v_simprocs_1368_ = l_Lean_ScopedEnvExtension_getState___redArg(v___x_1366_, v_simprocExt_1349_, v_env_1359_, v_asyncMode_1362_);
v___x_1369_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1369_, 0, v_base_1364_);
lean_ctor_set(v___x_1369_, 1, v_simpTheorems_1367_);
lean_ctor_set(v___x_1369_, 2, v_simprocs_1368_);
v___x_1370_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1370_, 0, v___x_1369_);
return v___x_1370_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_getGlobalRuleSetFromData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__14___redArg___boxed(lean_object* v_ext_1371_, lean_object* v_simpExt_1372_, lean_object* v_simprocExt_1373_, lean_object* v___y_1374_, lean_object* v___y_1375_){
_start:
{
lean_object* v_res_1376_; 
v_res_1376_ = lp_aesop_Aesop_Frontend_getGlobalRuleSetFromData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__14___redArg(v_ext_1371_, v_simpExt_1372_, v_simprocExt_1373_, v___y_1374_);
lean_dec(v___y_1374_);
lean_dec_ref(v_simprocExt_1373_);
lean_dec_ref(v_simpExt_1372_);
lean_dec_ref(v_ext_1371_);
return v_res_1376_;
}
}
static lean_object* _init_lp_aesop_Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2___closed__1(void){
_start:
{
lean_object* v___x_1378_; lean_object* v___x_1379_; 
v___x_1378_ = ((lean_object*)(lp_aesop_Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2___closed__0));
v___x_1379_ = l_Lean_stringToMessageData(v___x_1378_);
return v___x_1379_;
}
}
static lean_object* _init_lp_aesop_Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2___closed__3(void){
_start:
{
lean_object* v___x_1381_; lean_object* v___x_1382_; 
v___x_1381_ = ((lean_object*)(lp_aesop_Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2___closed__2));
v___x_1382_ = l_Lean_stringToMessageData(v___x_1381_);
return v___x_1382_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2(lean_object* v_rsName_1383_, lean_object* v_r_1384_, uint8_t v_kind_1385_, uint8_t v_checkNotExists_1386_, lean_object* v___y_1387_, lean_object* v___y_1388_){
_start:
{
lean_object* v___x_1390_; 
lean_inc(v_rsName_1383_);
v___x_1390_ = lp_aesop_Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__9(v_rsName_1383_, v___y_1387_, v___y_1388_);
if (lean_obj_tag(v___x_1390_) == 0)
{
lean_object* v_a_1391_; lean_object* v_snd_1392_; lean_object* v_snd_1393_; lean_object* v___x_1395_; uint8_t v_isShared_1396_; uint8_t v_isSharedCheck_1457_; 
v_a_1391_ = lean_ctor_get(v___x_1390_, 0);
lean_inc(v_a_1391_);
lean_dec_ref_known(v___x_1390_, 1);
v_snd_1392_ = lean_ctor_get(v_a_1391_, 1);
lean_inc(v_snd_1392_);
v_snd_1393_ = lean_ctor_get(v_snd_1392_, 1);
v_isSharedCheck_1457_ = !lean_is_exclusive(v_snd_1392_);
if (v_isSharedCheck_1457_ == 0)
{
lean_object* v_unused_1458_; 
v_unused_1458_ = lean_ctor_get(v_snd_1392_, 0);
lean_dec(v_unused_1458_);
v___x_1395_ = v_snd_1392_;
v_isShared_1396_ = v_isSharedCheck_1457_;
goto v_resetjp_1394_;
}
else
{
lean_inc(v_snd_1393_);
lean_dec(v_snd_1392_);
v___x_1395_ = lean_box(0);
v_isShared_1396_ = v_isSharedCheck_1457_;
goto v_resetjp_1394_;
}
v_resetjp_1394_:
{
lean_object* v_fst_1397_; lean_object* v___x_1399_; uint8_t v_isShared_1400_; uint8_t v_isSharedCheck_1455_; 
v_fst_1397_ = lean_ctor_get(v_a_1391_, 0);
v_isSharedCheck_1455_ = !lean_is_exclusive(v_a_1391_);
if (v_isSharedCheck_1455_ == 0)
{
lean_object* v_unused_1456_; 
v_unused_1456_ = lean_ctor_get(v_a_1391_, 1);
lean_dec(v_unused_1456_);
v___x_1399_ = v_a_1391_;
v_isShared_1400_ = v_isSharedCheck_1455_;
goto v_resetjp_1398_;
}
else
{
lean_inc(v_fst_1397_);
lean_dec(v_a_1391_);
v___x_1399_ = lean_box(0);
v_isShared_1400_ = v_isSharedCheck_1455_;
goto v_resetjp_1398_;
}
v_resetjp_1398_:
{
lean_object* v_fst_1401_; lean_object* v_snd_1402_; lean_object* v___x_1404_; uint8_t v_isShared_1405_; uint8_t v_isSharedCheck_1454_; 
v_fst_1401_ = lean_ctor_get(v_snd_1393_, 0);
v_snd_1402_ = lean_ctor_get(v_snd_1393_, 1);
v_isSharedCheck_1454_ = !lean_is_exclusive(v_snd_1393_);
if (v_isSharedCheck_1454_ == 0)
{
v___x_1404_ = v_snd_1393_;
v_isShared_1405_ = v_isSharedCheck_1454_;
goto v_resetjp_1403_;
}
else
{
lean_inc(v_snd_1402_);
lean_inc(v_fst_1401_);
lean_dec(v_snd_1393_);
v___x_1404_ = lean_box(0);
v_isShared_1405_ = v_isSharedCheck_1454_;
goto v_resetjp_1403_;
}
v_resetjp_1403_:
{
lean_object* v___y_1407_; lean_object* v___y_1408_; 
if (v_checkNotExists_1386_ == 0)
{
lean_del_object(v___x_1404_);
lean_dec(v_snd_1402_);
lean_del_object(v___x_1399_);
lean_del_object(v___x_1395_);
lean_dec(v_rsName_1383_);
v___y_1407_ = v___y_1387_;
v___y_1408_ = v___y_1388_;
goto v___jp_1406_;
}
else
{
lean_object* v_snd_1425_; lean_object* v___x_1427_; uint8_t v_isShared_1428_; uint8_t v_isSharedCheck_1452_; 
v_snd_1425_ = lean_ctor_get(v_snd_1402_, 1);
v_isSharedCheck_1452_ = !lean_is_exclusive(v_snd_1402_);
if (v_isSharedCheck_1452_ == 0)
{
lean_object* v_unused_1453_; 
v_unused_1453_ = lean_ctor_get(v_snd_1402_, 0);
lean_dec(v_unused_1453_);
v___x_1427_ = v_snd_1402_;
v_isShared_1428_ = v_isSharedCheck_1452_;
goto v_resetjp_1426_;
}
else
{
lean_inc(v_snd_1425_);
lean_dec(v_snd_1402_);
v___x_1427_ = lean_box(0);
v_isShared_1428_ = v_isSharedCheck_1452_;
goto v_resetjp_1426_;
}
v_resetjp_1426_:
{
lean_object* v___x_1429_; lean_object* v_a_1430_; lean_object* v___x_1431_; uint8_t v___x_1432_; 
v___x_1429_ = lp_aesop_Aesop_Frontend_getGlobalRuleSetFromData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__14___redArg(v_fst_1397_, v_fst_1401_, v_snd_1425_, v___y_1388_);
lean_dec(v_snd_1425_);
v_a_1430_ = lean_ctor_get(v___x_1429_, 0);
lean_inc(v_a_1430_);
lean_dec_ref(v___x_1429_);
v___x_1431_ = lp_aesop_Aesop_GlobalRuleSetMember_name(v_r_1384_);
lean_inc_ref(v___x_1431_);
v___x_1432_ = lp_aesop_Aesop_GlobalRuleSet_contains(v_a_1430_, v___x_1431_);
lean_dec(v_a_1430_);
if (v___x_1432_ == 0)
{
lean_dec_ref(v___x_1431_);
lean_del_object(v___x_1427_);
lean_del_object(v___x_1404_);
lean_del_object(v___x_1399_);
lean_del_object(v___x_1395_);
lean_dec(v_rsName_1383_);
v___y_1407_ = v___y_1387_;
v___y_1408_ = v___y_1388_;
goto v___jp_1406_;
}
else
{
lean_object* v_name_1433_; lean_object* v___x_1434_; lean_object* v___x_1435_; lean_object* v___x_1437_; 
lean_dec(v_fst_1401_);
lean_dec(v_fst_1397_);
lean_dec_ref(v_r_1384_);
v_name_1433_ = lean_ctor_get(v___x_1431_, 0);
lean_inc(v_name_1433_);
lean_dec_ref(v___x_1431_);
v___x_1434_ = lean_obj_once(&lp_aesop_Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2___closed__1, &lp_aesop_Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2___closed__1_once, _init_lp_aesop_Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2___closed__1);
v___x_1435_ = l_Lean_MessageData_ofName(v_name_1433_);
if (v_isShared_1428_ == 0)
{
lean_ctor_set_tag(v___x_1427_, 7);
lean_ctor_set(v___x_1427_, 1, v___x_1435_);
lean_ctor_set(v___x_1427_, 0, v___x_1434_);
v___x_1437_ = v___x_1427_;
goto v_reusejp_1436_;
}
else
{
lean_object* v_reuseFailAlloc_1451_; 
v_reuseFailAlloc_1451_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1451_, 0, v___x_1434_);
lean_ctor_set(v_reuseFailAlloc_1451_, 1, v___x_1435_);
v___x_1437_ = v_reuseFailAlloc_1451_;
goto v_reusejp_1436_;
}
v_reusejp_1436_:
{
lean_object* v___x_1438_; lean_object* v___x_1440_; 
v___x_1438_ = lean_obj_once(&lp_aesop_Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2___closed__3, &lp_aesop_Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2___closed__3_once, _init_lp_aesop_Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2___closed__3);
if (v_isShared_1405_ == 0)
{
lean_ctor_set_tag(v___x_1404_, 7);
lean_ctor_set(v___x_1404_, 1, v___x_1438_);
lean_ctor_set(v___x_1404_, 0, v___x_1437_);
v___x_1440_ = v___x_1404_;
goto v_reusejp_1439_;
}
else
{
lean_object* v_reuseFailAlloc_1450_; 
v_reuseFailAlloc_1450_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1450_, 0, v___x_1437_);
lean_ctor_set(v_reuseFailAlloc_1450_, 1, v___x_1438_);
v___x_1440_ = v_reuseFailAlloc_1450_;
goto v_reusejp_1439_;
}
v_reusejp_1439_:
{
lean_object* v___x_1441_; lean_object* v___x_1443_; 
v___x_1441_ = l_Lean_MessageData_ofName(v_rsName_1383_);
if (v_isShared_1396_ == 0)
{
lean_ctor_set_tag(v___x_1395_, 7);
lean_ctor_set(v___x_1395_, 1, v___x_1441_);
lean_ctor_set(v___x_1395_, 0, v___x_1440_);
v___x_1443_ = v___x_1395_;
goto v_reusejp_1442_;
}
else
{
lean_object* v_reuseFailAlloc_1449_; 
v_reuseFailAlloc_1449_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1449_, 0, v___x_1440_);
lean_ctor_set(v_reuseFailAlloc_1449_, 1, v___x_1441_);
v___x_1443_ = v_reuseFailAlloc_1449_;
goto v_reusejp_1442_;
}
v_reusejp_1442_:
{
lean_object* v___x_1444_; lean_object* v___x_1446_; 
v___x_1444_ = lean_obj_once(&lp_aesop_Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0___closed__1, &lp_aesop_Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0___closed__1_once, _init_lp_aesop_Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0___closed__1);
if (v_isShared_1400_ == 0)
{
lean_ctor_set_tag(v___x_1399_, 7);
lean_ctor_set(v___x_1399_, 1, v___x_1444_);
lean_ctor_set(v___x_1399_, 0, v___x_1443_);
v___x_1446_ = v___x_1399_;
goto v_reusejp_1445_;
}
else
{
lean_object* v_reuseFailAlloc_1448_; 
v_reuseFailAlloc_1448_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1448_, 0, v___x_1443_);
lean_ctor_set(v_reuseFailAlloc_1448_, 1, v___x_1444_);
v___x_1446_ = v_reuseFailAlloc_1448_;
goto v_reusejp_1445_;
}
v_reusejp_1445_:
{
lean_object* v___x_1447_; 
v___x_1447_ = lp_aesop_Lean_throwError___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0_spec__0___redArg(v___x_1446_, v___y_1387_, v___y_1388_);
return v___x_1447_;
}
}
}
}
}
}
}
v___jp_1406_:
{
if (lean_obj_tag(v_r_1384_) == 0)
{
lean_object* v_m_1409_; lean_object* v___x_1410_; 
lean_dec(v_fst_1401_);
v_m_1409_ = lean_ctor_get(v_r_1384_, 0);
lean_inc_ref(v_m_1409_);
lean_dec_ref_known(v_r_1384_, 1);
v___x_1410_ = lp_aesop_Lean_ScopedEnvExtension_add___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__10___redArg(v_fst_1397_, v_m_1409_, v_kind_1385_, v___y_1407_, v___y_1408_);
return v___x_1410_;
}
else
{
lean_object* v_e_1411_; lean_object* v_entries_1412_; lean_object* v___x_1413_; size_t v_sz_1414_; size_t v___x_1415_; lean_object* v___x_1416_; 
lean_dec(v_fst_1397_);
v_e_1411_ = lean_ctor_get(v_r_1384_, 0);
lean_inc_ref(v_e_1411_);
lean_dec_ref_known(v_r_1384_, 1);
v_entries_1412_ = lean_ctor_get(v_e_1411_, 1);
lean_inc_ref(v_entries_1412_);
lean_dec_ref(v_e_1411_);
v___x_1413_ = lean_box(0);
v_sz_1414_ = lean_array_size(v_entries_1412_);
v___x_1415_ = ((size_t)0ULL);
v___x_1416_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__13(v_fst_1401_, v_kind_1385_, v_entries_1412_, v_sz_1414_, v___x_1415_, v___x_1413_, v___y_1407_, v___y_1408_);
lean_dec_ref(v_entries_1412_);
if (lean_obj_tag(v___x_1416_) == 0)
{
lean_object* v___x_1418_; uint8_t v_isShared_1419_; uint8_t v_isSharedCheck_1423_; 
v_isSharedCheck_1423_ = !lean_is_exclusive(v___x_1416_);
if (v_isSharedCheck_1423_ == 0)
{
lean_object* v_unused_1424_; 
v_unused_1424_ = lean_ctor_get(v___x_1416_, 0);
lean_dec(v_unused_1424_);
v___x_1418_ = v___x_1416_;
v_isShared_1419_ = v_isSharedCheck_1423_;
goto v_resetjp_1417_;
}
else
{
lean_dec(v___x_1416_);
v___x_1418_ = lean_box(0);
v_isShared_1419_ = v_isSharedCheck_1423_;
goto v_resetjp_1417_;
}
v_resetjp_1417_:
{
lean_object* v___x_1421_; 
if (v_isShared_1419_ == 0)
{
lean_ctor_set(v___x_1418_, 0, v___x_1413_);
v___x_1421_ = v___x_1418_;
goto v_reusejp_1420_;
}
else
{
lean_object* v_reuseFailAlloc_1422_; 
v_reuseFailAlloc_1422_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1422_, 0, v___x_1413_);
v___x_1421_ = v_reuseFailAlloc_1422_;
goto v_reusejp_1420_;
}
v_reusejp_1420_:
{
return v___x_1421_;
}
}
}
else
{
return v___x_1416_;
}
}
}
}
}
}
}
else
{
lean_object* v_a_1459_; lean_object* v___x_1461_; uint8_t v_isShared_1462_; uint8_t v_isSharedCheck_1466_; 
lean_dec_ref(v_r_1384_);
lean_dec(v_rsName_1383_);
v_a_1459_ = lean_ctor_get(v___x_1390_, 0);
v_isSharedCheck_1466_ = !lean_is_exclusive(v___x_1390_);
if (v_isSharedCheck_1466_ == 0)
{
v___x_1461_ = v___x_1390_;
v_isShared_1462_ = v_isSharedCheck_1466_;
goto v_resetjp_1460_;
}
else
{
lean_inc(v_a_1459_);
lean_dec(v___x_1390_);
v___x_1461_ = lean_box(0);
v_isShared_1462_ = v_isSharedCheck_1466_;
goto v_resetjp_1460_;
}
v_resetjp_1460_:
{
lean_object* v___x_1464_; 
if (v_isShared_1462_ == 0)
{
v___x_1464_ = v___x_1461_;
goto v_reusejp_1463_;
}
else
{
lean_object* v_reuseFailAlloc_1465_; 
v_reuseFailAlloc_1465_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1465_, 0, v_a_1459_);
v___x_1464_ = v_reuseFailAlloc_1465_;
goto v_reusejp_1463_;
}
v_reusejp_1463_:
{
return v___x_1464_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2___boxed(lean_object* v_rsName_1467_, lean_object* v_r_1468_, lean_object* v_kind_1469_, lean_object* v_checkNotExists_1470_, lean_object* v___y_1471_, lean_object* v___y_1472_, lean_object* v___y_1473_){
_start:
{
uint8_t v_kind_boxed_1474_; uint8_t v_checkNotExists_boxed_1475_; lean_object* v_res_1476_; 
v_kind_boxed_1474_ = lean_unbox(v_kind_1469_);
v_checkNotExists_boxed_1475_ = lean_unbox(v_checkNotExists_1470_);
v_res_1476_ = lp_aesop_Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2(v_rsName_1467_, v_r_1468_, v_kind_boxed_1474_, v_checkNotExists_boxed_1475_, v___y_1471_, v___y_1472_);
lean_dec(v___y_1472_);
lean_dec_ref(v___y_1471_);
return v_res_1476_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__3(lean_object* v_fst_1477_, uint8_t v_attrKind_1478_, lean_object* v_as_1479_, size_t v_sz_1480_, size_t v_i_1481_, lean_object* v_b_1482_, lean_object* v___y_1483_, lean_object* v___y_1484_){
_start:
{
uint8_t v___x_1486_; 
v___x_1486_ = lean_usize_dec_lt(v_i_1481_, v_sz_1480_);
if (v___x_1486_ == 0)
{
lean_object* v___x_1487_; 
lean_dec_ref(v_fst_1477_);
v___x_1487_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1487_, 0, v_b_1482_);
return v___x_1487_;
}
else
{
lean_object* v_a_1488_; lean_object* v___x_1489_; 
v_a_1488_ = lean_array_uget_borrowed(v_as_1479_, v_i_1481_);
lean_inc_ref(v_fst_1477_);
lean_inc(v_a_1488_);
v___x_1489_ = lp_aesop_Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2(v_a_1488_, v_fst_1477_, v_attrKind_1478_, v___x_1486_, v___y_1483_, v___y_1484_);
if (lean_obj_tag(v___x_1489_) == 0)
{
lean_object* v___x_1490_; size_t v___x_1491_; size_t v___x_1492_; 
lean_dec_ref_known(v___x_1489_, 1);
v___x_1490_ = lean_box(0);
v___x_1491_ = ((size_t)1ULL);
v___x_1492_ = lean_usize_add(v_i_1481_, v___x_1491_);
v_i_1481_ = v___x_1492_;
v_b_1482_ = v___x_1490_;
goto _start;
}
else
{
lean_dec_ref(v_fst_1477_);
return v___x_1489_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__3___boxed(lean_object* v_fst_1494_, lean_object* v_attrKind_1495_, lean_object* v_as_1496_, lean_object* v_sz_1497_, lean_object* v_i_1498_, lean_object* v_b_1499_, lean_object* v___y_1500_, lean_object* v___y_1501_, lean_object* v___y_1502_){
_start:
{
uint8_t v_attrKind_boxed_1503_; size_t v_sz_boxed_1504_; size_t v_i_boxed_1505_; lean_object* v_res_1506_; 
v_attrKind_boxed_1503_ = lean_unbox(v_attrKind_1495_);
v_sz_boxed_1504_ = lean_unbox_usize(v_sz_1497_);
lean_dec(v_sz_1497_);
v_i_boxed_1505_ = lean_unbox_usize(v_i_1498_);
lean_dec(v_i_1498_);
v_res_1506_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__3(v_fst_1494_, v_attrKind_boxed_1503_, v_as_1496_, v_sz_boxed_1504_, v_i_boxed_1505_, v_b_1499_, v___y_1500_, v___y_1501_);
lean_dec(v___y_1501_);
lean_dec_ref(v___y_1500_);
lean_dec_ref(v_as_1496_);
return v_res_1506_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__4(uint8_t v_attrKind_1507_, lean_object* v_as_1508_, size_t v_sz_1509_, size_t v_i_1510_, lean_object* v_b_1511_, lean_object* v___y_1512_, lean_object* v___y_1513_){
_start:
{
uint8_t v___x_1515_; 
v___x_1515_ = lean_usize_dec_lt(v_i_1510_, v_sz_1509_);
if (v___x_1515_ == 0)
{
lean_object* v___x_1516_; 
v___x_1516_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1516_, 0, v_b_1511_);
return v___x_1516_;
}
else
{
lean_object* v_a_1517_; lean_object* v_fst_1518_; lean_object* v_snd_1519_; lean_object* v___x_1520_; size_t v_sz_1521_; size_t v___x_1522_; lean_object* v___x_1523_; 
v_a_1517_ = lean_array_uget_borrowed(v_as_1508_, v_i_1510_);
v_fst_1518_ = lean_ctor_get(v_a_1517_, 0);
v_snd_1519_ = lean_ctor_get(v_a_1517_, 1);
v___x_1520_ = lean_box(0);
v_sz_1521_ = lean_array_size(v_snd_1519_);
v___x_1522_ = ((size_t)0ULL);
lean_inc(v_fst_1518_);
v___x_1523_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__3(v_fst_1518_, v_attrKind_1507_, v_snd_1519_, v_sz_1521_, v___x_1522_, v___x_1520_, v___y_1512_, v___y_1513_);
if (lean_obj_tag(v___x_1523_) == 0)
{
size_t v___x_1524_; size_t v___x_1525_; 
lean_dec_ref_known(v___x_1523_, 1);
v___x_1524_ = ((size_t)1ULL);
v___x_1525_ = lean_usize_add(v_i_1510_, v___x_1524_);
v_i_1510_ = v___x_1525_;
v_b_1511_ = v___x_1520_;
goto _start;
}
else
{
return v___x_1523_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__4___boxed(lean_object* v_attrKind_1527_, lean_object* v_as_1528_, lean_object* v_sz_1529_, lean_object* v_i_1530_, lean_object* v_b_1531_, lean_object* v___y_1532_, lean_object* v___y_1533_, lean_object* v___y_1534_){
_start:
{
uint8_t v_attrKind_boxed_1535_; size_t v_sz_boxed_1536_; size_t v_i_boxed_1537_; lean_object* v_res_1538_; 
v_attrKind_boxed_1535_ = lean_unbox(v_attrKind_1527_);
v_sz_boxed_1536_ = lean_unbox_usize(v_sz_1529_);
lean_dec(v_sz_1529_);
v_i_boxed_1537_ = lean_unbox_usize(v_i_1530_);
lean_dec(v_i_1530_);
v_res_1538_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__4(v_attrKind_boxed_1535_, v_as_1528_, v_sz_boxed_1536_, v_i_boxed_1537_, v_b_1531_, v___y_1532_, v___y_1533_);
lean_dec(v___y_1533_);
lean_dec_ref(v___y_1532_);
lean_dec_ref(v_as_1528_);
return v_res_1538_;
}
}
static uint64_t _init_lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___lam__3___closed__2_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_1547_; uint64_t v___x_1548_; 
v___x_1547_ = ((lean_object*)(lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___lam__3___closed__1_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2_));
v___x_1548_ = l___private_Lean_Meta_Basic_0__Lean_Meta_Config_toKey(v___x_1547_);
return v___x_1548_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___lam__3___closed__3_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2_(void){
_start:
{
uint64_t v___x_1549_; lean_object* v___x_1550_; lean_object* v___x_1551_; 
v___x_1549_ = lean_uint64_once(&lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___lam__3___closed__2_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2_, &lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___lam__3___closed__2_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__once, _init_lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___lam__3___closed__2_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2_);
v___x_1550_ = ((lean_object*)(lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___lam__3___closed__1_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2_));
v___x_1551_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v___x_1551_, 0, v___x_1550_);
lean_ctor_set_uint64(v___x_1551_, sizeof(void*)*1, v___x_1549_);
return v___x_1551_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___lam__3___closed__4_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_1552_; 
v___x_1552_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_1552_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___lam__3___closed__5_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_1553_; lean_object* v___x_1554_; 
v___x_1553_ = lean_obj_once(&lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___lam__3___closed__4_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2_, &lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___lam__3___closed__4_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__once, _init_lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___lam__3___closed__4_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2_);
v___x_1554_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1554_, 0, v___x_1553_);
return v___x_1554_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___lam__3___closed__6_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_1555_; lean_object* v___x_1556_; lean_object* v___x_1557_; lean_object* v___x_1558_; 
v___x_1555_ = lean_box(1);
v___x_1556_ = lean_obj_once(&lp_aesop_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0_spec__0_spec__1___closed__4, &lp_aesop_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0_spec__0_spec__1___closed__4_once, _init_lp_aesop_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0_spec__0_spec__1___closed__4);
v___x_1557_ = lean_obj_once(&lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___lam__3___closed__5_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2_, &lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___lam__3___closed__5_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__once, _init_lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___lam__3___closed__5_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2_);
v___x_1558_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1558_, 0, v___x_1557_);
lean_ctor_set(v___x_1558_, 1, v___x_1556_);
lean_ctor_set(v___x_1558_, 2, v___x_1555_);
return v___x_1558_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___lam__3___closed__7_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2_(void){
_start:
{
uint8_t v___x_1559_; lean_object* v___x_1560_; lean_object* v___x_1561_; lean_object* v___x_1562_; lean_object* v___x_1563_; lean_object* v___x_1564_; uint8_t v___x_1565_; lean_object* v___x_1566_; lean_object* v___x_1567_; 
v___x_1559_ = 1;
v___x_1560_ = lean_unsigned_to_nat(0u);
v___x_1561_ = lean_box(0);
v___x_1562_ = ((lean_object*)(lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___lam__3___closed__0_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2_));
v___x_1563_ = lean_obj_once(&lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___lam__3___closed__6_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2_, &lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___lam__3___closed__6_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__once, _init_lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___lam__3___closed__6_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2_);
v___x_1564_ = lean_box(1);
v___x_1565_ = 0;
v___x_1566_ = lean_obj_once(&lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___lam__3___closed__3_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2_, &lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___lam__3___closed__3_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__once, _init_lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___lam__3___closed__3_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2_);
v___x_1567_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v___x_1567_, 0, v___x_1566_);
lean_ctor_set(v___x_1567_, 1, v___x_1564_);
lean_ctor_set(v___x_1567_, 2, v___x_1563_);
lean_ctor_set(v___x_1567_, 3, v___x_1562_);
lean_ctor_set(v___x_1567_, 4, v___x_1561_);
lean_ctor_set(v___x_1567_, 5, v___x_1560_);
lean_ctor_set(v___x_1567_, 6, v___x_1561_);
lean_ctor_set_uint8(v___x_1567_, sizeof(void*)*7, v___x_1565_);
lean_ctor_set_uint8(v___x_1567_, sizeof(void*)*7 + 1, v___x_1565_);
lean_ctor_set_uint8(v___x_1567_, sizeof(void*)*7 + 2, v___x_1565_);
lean_ctor_set_uint8(v___x_1567_, sizeof(void*)*7 + 3, v___x_1559_);
return v___x_1567_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___lam__3___closed__8_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_1568_; lean_object* v___x_1569_; lean_object* v___x_1570_; 
v___x_1568_ = lean_obj_once(&lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___lam__3___closed__5_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2_, &lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___lam__3___closed__5_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__once, _init_lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___lam__3___closed__5_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2_);
v___x_1569_ = lean_unsigned_to_nat(0u);
v___x_1570_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v___x_1570_, 0, v___x_1569_);
lean_ctor_set(v___x_1570_, 1, v___x_1569_);
lean_ctor_set(v___x_1570_, 2, v___x_1569_);
lean_ctor_set(v___x_1570_, 3, v___x_1569_);
lean_ctor_set(v___x_1570_, 4, v___x_1568_);
lean_ctor_set(v___x_1570_, 5, v___x_1568_);
lean_ctor_set(v___x_1570_, 6, v___x_1568_);
lean_ctor_set(v___x_1570_, 7, v___x_1568_);
lean_ctor_set(v___x_1570_, 8, v___x_1568_);
lean_ctor_set(v___x_1570_, 9, v___x_1568_);
return v___x_1570_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___lam__3___closed__9_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_1571_; lean_object* v___x_1572_; 
v___x_1571_ = lean_obj_once(&lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___lam__3___closed__5_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2_, &lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___lam__3___closed__5_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__once, _init_lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___lam__3___closed__5_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2_);
v___x_1572_ = lean_alloc_ctor(0, 6, 0);
lean_ctor_set(v___x_1572_, 0, v___x_1571_);
lean_ctor_set(v___x_1572_, 1, v___x_1571_);
lean_ctor_set(v___x_1572_, 2, v___x_1571_);
lean_ctor_set(v___x_1572_, 3, v___x_1571_);
lean_ctor_set(v___x_1572_, 4, v___x_1571_);
lean_ctor_set(v___x_1572_, 5, v___x_1571_);
return v___x_1572_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___lam__3___closed__10_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_1573_; lean_object* v___x_1574_; 
v___x_1573_ = lean_obj_once(&lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___lam__3___closed__5_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2_, &lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___lam__3___closed__5_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__once, _init_lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___lam__3___closed__5_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2_);
v___x_1574_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_1574_, 0, v___x_1573_);
lean_ctor_set(v___x_1574_, 1, v___x_1573_);
lean_ctor_set(v___x_1574_, 2, v___x_1573_);
lean_ctor_set(v___x_1574_, 3, v___x_1573_);
lean_ctor_set(v___x_1574_, 4, v___x_1573_);
return v___x_1574_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___lam__3___closed__11_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_1575_; lean_object* v___x_1576_; lean_object* v___x_1577_; lean_object* v___x_1578_; lean_object* v___x_1579_; lean_object* v___x_1580_; 
v___x_1575_ = lean_obj_once(&lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___lam__3___closed__10_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2_, &lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___lam__3___closed__10_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__once, _init_lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___lam__3___closed__10_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2_);
v___x_1576_ = lean_obj_once(&lp_aesop_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0_spec__0_spec__1___closed__4, &lp_aesop_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0_spec__0_spec__1___closed__4_once, _init_lp_aesop_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0_spec__0_spec__1___closed__4);
v___x_1577_ = lean_box(1);
v___x_1578_ = lean_obj_once(&lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___lam__3___closed__9_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2_, &lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___lam__3___closed__9_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__once, _init_lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___lam__3___closed__9_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2_);
v___x_1579_ = lean_obj_once(&lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___lam__3___closed__8_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2_, &lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___lam__3___closed__8_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__once, _init_lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___lam__3___closed__8_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2_);
v___x_1580_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_1580_, 0, v___x_1579_);
lean_ctor_set(v___x_1580_, 1, v___x_1578_);
lean_ctor_set(v___x_1580_, 2, v___x_1577_);
lean_ctor_set(v___x_1580_, 3, v___x_1576_);
lean_ctor_set(v___x_1580_, 4, v___x_1575_);
return v___x_1580_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___lam__3_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2_(lean_object* v___f_1584_, lean_object* v_decl_1585_, lean_object* v_stx_1586_, uint8_t v_attrKind_1587_, lean_object* v___y_1588_, lean_object* v___y_1589_){
_start:
{
lean_object* v___x_1591_; lean_object* v___x_1592_; uint8_t v___x_1593_; lean_object* v___x_1594_; uint8_t v___x_1595_; lean_object* v___x_1596_; lean_object* v___x_1597_; lean_object* v___x_1598_; lean_object* v___x_1599_; lean_object* v___x_1600_; lean_object* v_fileName_1601_; lean_object* v_fileMap_1602_; lean_object* v_options_1603_; lean_object* v_currRecDepth_1604_; lean_object* v_maxRecDepth_1605_; lean_object* v_ref_1606_; lean_object* v_currNamespace_1607_; lean_object* v_openDecls_1608_; lean_object* v_initHeartbeats_1609_; lean_object* v_maxHeartbeats_1610_; lean_object* v_quotContext_1611_; lean_object* v_currMacroScope_1612_; uint8_t v_diag_1613_; lean_object* v_cancelTk_x3f_1614_; uint8_t v_suppressElabErrors_1615_; lean_object* v_inheritedTraceOptions_1616_; lean_object* v___f_1617_; lean_object* v_ref_1618_; lean_object* v___x_1619_; lean_object* v___x_1620_; lean_object* v___x_1621_; 
v___x_1591_ = lean_box(0);
v___x_1592_ = lean_box(0);
v___x_1593_ = 1;
v___x_1594_ = lean_box(1);
v___x_1595_ = 0;
v___x_1596_ = ((lean_object*)(lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___lam__3___closed__0_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2_));
v___x_1597_ = lean_alloc_ctor(0, 8, 11);
lean_ctor_set(v___x_1597_, 0, v___x_1591_);
lean_ctor_set(v___x_1597_, 1, v___x_1592_);
lean_ctor_set(v___x_1597_, 2, v___x_1591_);
lean_ctor_set(v___x_1597_, 3, v___f_1584_);
lean_ctor_set(v___x_1597_, 4, v___x_1594_);
lean_ctor_set(v___x_1597_, 5, v___x_1594_);
lean_ctor_set(v___x_1597_, 6, v___x_1591_);
lean_ctor_set(v___x_1597_, 7, v___x_1596_);
lean_ctor_set_uint8(v___x_1597_, sizeof(void*)*8, v___x_1593_);
lean_ctor_set_uint8(v___x_1597_, sizeof(void*)*8 + 1, v___x_1593_);
lean_ctor_set_uint8(v___x_1597_, sizeof(void*)*8 + 2, v___x_1593_);
lean_ctor_set_uint8(v___x_1597_, sizeof(void*)*8 + 3, v___x_1593_);
lean_ctor_set_uint8(v___x_1597_, sizeof(void*)*8 + 4, v___x_1595_);
lean_ctor_set_uint8(v___x_1597_, sizeof(void*)*8 + 5, v___x_1595_);
lean_ctor_set_uint8(v___x_1597_, sizeof(void*)*8 + 6, v___x_1595_);
lean_ctor_set_uint8(v___x_1597_, sizeof(void*)*8 + 7, v___x_1595_);
lean_ctor_set_uint8(v___x_1597_, sizeof(void*)*8 + 8, v___x_1593_);
lean_ctor_set_uint8(v___x_1597_, sizeof(void*)*8 + 9, v___x_1595_);
lean_ctor_set_uint8(v___x_1597_, sizeof(void*)*8 + 10, v___x_1593_);
v___x_1598_ = lean_obj_once(&lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___lam__3___closed__7_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2_, &lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___lam__3___closed__7_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__once, _init_lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___lam__3___closed__7_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2_);
v___x_1599_ = lean_obj_once(&lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___lam__3___closed__11_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2_, &lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___lam__3___closed__11_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__once, _init_lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___lam__3___closed__11_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2_);
v___x_1600_ = lean_st_mk_ref(v___x_1599_);
v_fileName_1601_ = lean_ctor_get(v___y_1588_, 0);
v_fileMap_1602_ = lean_ctor_get(v___y_1588_, 1);
v_options_1603_ = lean_ctor_get(v___y_1588_, 2);
v_currRecDepth_1604_ = lean_ctor_get(v___y_1588_, 3);
v_maxRecDepth_1605_ = lean_ctor_get(v___y_1588_, 4);
v_ref_1606_ = lean_ctor_get(v___y_1588_, 5);
v_currNamespace_1607_ = lean_ctor_get(v___y_1588_, 6);
v_openDecls_1608_ = lean_ctor_get(v___y_1588_, 7);
v_initHeartbeats_1609_ = lean_ctor_get(v___y_1588_, 8);
v_maxHeartbeats_1610_ = lean_ctor_get(v___y_1588_, 9);
v_quotContext_1611_ = lean_ctor_get(v___y_1588_, 10);
v_currMacroScope_1612_ = lean_ctor_get(v___y_1588_, 11);
v_diag_1613_ = lean_ctor_get_uint8(v___y_1588_, sizeof(void*)*14);
v_cancelTk_x3f_1614_ = lean_ctor_get(v___y_1588_, 12);
v_suppressElabErrors_1615_ = lean_ctor_get_uint8(v___y_1588_, sizeof(void*)*14 + 1);
v_inheritedTraceOptions_1616_ = lean_ctor_get(v___y_1588_, 13);
lean_inc(v_stx_1586_);
v___f_1617_ = lean_alloc_closure((void*)(lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___lam__2_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2____boxed), 9, 2);
lean_closure_set(v___f_1617_, 0, v_stx_1586_);
lean_closure_set(v___f_1617_, 1, v_decl_1585_);
v_ref_1618_ = l_Lean_replaceRef(v_stx_1586_, v_ref_1606_);
lean_dec(v_stx_1586_);
lean_inc_ref(v_inheritedTraceOptions_1616_);
lean_inc(v_cancelTk_x3f_1614_);
lean_inc(v_currMacroScope_1612_);
lean_inc(v_quotContext_1611_);
lean_inc(v_maxHeartbeats_1610_);
lean_inc(v_initHeartbeats_1609_);
lean_inc(v_openDecls_1608_);
lean_inc(v_currNamespace_1607_);
lean_inc(v_maxRecDepth_1605_);
lean_inc(v_currRecDepth_1604_);
lean_inc_ref(v_options_1603_);
lean_inc_ref(v_fileMap_1602_);
lean_inc_ref(v_fileName_1601_);
v___x_1619_ = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(v___x_1619_, 0, v_fileName_1601_);
lean_ctor_set(v___x_1619_, 1, v_fileMap_1602_);
lean_ctor_set(v___x_1619_, 2, v_options_1603_);
lean_ctor_set(v___x_1619_, 3, v_currRecDepth_1604_);
lean_ctor_set(v___x_1619_, 4, v_maxRecDepth_1605_);
lean_ctor_set(v___x_1619_, 5, v_ref_1618_);
lean_ctor_set(v___x_1619_, 6, v_currNamespace_1607_);
lean_ctor_set(v___x_1619_, 7, v_openDecls_1608_);
lean_ctor_set(v___x_1619_, 8, v_initHeartbeats_1609_);
lean_ctor_set(v___x_1619_, 9, v_maxHeartbeats_1610_);
lean_ctor_set(v___x_1619_, 10, v_quotContext_1611_);
lean_ctor_set(v___x_1619_, 11, v_currMacroScope_1612_);
lean_ctor_set(v___x_1619_, 12, v_cancelTk_x3f_1614_);
lean_ctor_set(v___x_1619_, 13, v_inheritedTraceOptions_1616_);
lean_ctor_set_uint8(v___x_1619_, sizeof(void*)*14, v_diag_1613_);
lean_ctor_set_uint8(v___x_1619_, sizeof(void*)*14 + 1, v_suppressElabErrors_1615_);
v___x_1620_ = ((lean_object*)(lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___lam__3___closed__12_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2_));
v___x_1621_ = l_Lean_Elab_Term_TermElabM_run___redArg(v___f_1617_, v___x_1597_, v___x_1620_, v___x_1598_, v___x_1600_, v___x_1619_, v___y_1589_);
if (lean_obj_tag(v___x_1621_) == 0)
{
lean_object* v_a_1622_; lean_object* v___x_1623_; lean_object* v_fst_1624_; lean_object* v___x_1625_; size_t v_sz_1626_; size_t v___x_1627_; lean_object* v___x_1628_; 
v_a_1622_ = lean_ctor_get(v___x_1621_, 0);
lean_inc(v_a_1622_);
lean_dec_ref_known(v___x_1621_, 1);
v___x_1623_ = lean_st_ref_get(v___x_1600_);
lean_dec(v___x_1600_);
lean_dec(v___x_1623_);
v_fst_1624_ = lean_ctor_get(v_a_1622_, 0);
lean_inc(v_fst_1624_);
lean_dec(v_a_1622_);
v___x_1625_ = lean_box(0);
v_sz_1626_ = lean_array_size(v_fst_1624_);
v___x_1627_ = ((size_t)0ULL);
v___x_1628_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__4(v_attrKind_1587_, v_fst_1624_, v_sz_1626_, v___x_1627_, v___x_1625_, v___x_1619_, v___y_1589_);
lean_dec_ref_known(v___x_1619_, 14);
lean_dec(v_fst_1624_);
if (lean_obj_tag(v___x_1628_) == 0)
{
lean_object* v___x_1630_; uint8_t v_isShared_1631_; uint8_t v_isSharedCheck_1635_; 
v_isSharedCheck_1635_ = !lean_is_exclusive(v___x_1628_);
if (v_isSharedCheck_1635_ == 0)
{
lean_object* v_unused_1636_; 
v_unused_1636_ = lean_ctor_get(v___x_1628_, 0);
lean_dec(v_unused_1636_);
v___x_1630_ = v___x_1628_;
v_isShared_1631_ = v_isSharedCheck_1635_;
goto v_resetjp_1629_;
}
else
{
lean_dec(v___x_1628_);
v___x_1630_ = lean_box(0);
v_isShared_1631_ = v_isSharedCheck_1635_;
goto v_resetjp_1629_;
}
v_resetjp_1629_:
{
lean_object* v___x_1633_; 
if (v_isShared_1631_ == 0)
{
lean_ctor_set(v___x_1630_, 0, v___x_1625_);
v___x_1633_ = v___x_1630_;
goto v_reusejp_1632_;
}
else
{
lean_object* v_reuseFailAlloc_1634_; 
v_reuseFailAlloc_1634_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1634_, 0, v___x_1625_);
v___x_1633_ = v_reuseFailAlloc_1634_;
goto v_reusejp_1632_;
}
v_reusejp_1632_:
{
return v___x_1633_;
}
}
}
else
{
return v___x_1628_;
}
}
else
{
lean_object* v_a_1637_; lean_object* v___x_1639_; uint8_t v_isShared_1640_; uint8_t v_isSharedCheck_1644_; 
lean_dec_ref_known(v___x_1619_, 14);
lean_dec(v___x_1600_);
v_a_1637_ = lean_ctor_get(v___x_1621_, 0);
v_isSharedCheck_1644_ = !lean_is_exclusive(v___x_1621_);
if (v_isSharedCheck_1644_ == 0)
{
v___x_1639_ = v___x_1621_;
v_isShared_1640_ = v_isSharedCheck_1644_;
goto v_resetjp_1638_;
}
else
{
lean_inc(v_a_1637_);
lean_dec(v___x_1621_);
v___x_1639_ = lean_box(0);
v_isShared_1640_ = v_isSharedCheck_1644_;
goto v_resetjp_1638_;
}
v_resetjp_1638_:
{
lean_object* v___x_1642_; 
if (v_isShared_1640_ == 0)
{
v___x_1642_ = v___x_1639_;
goto v_reusejp_1641_;
}
else
{
lean_object* v_reuseFailAlloc_1643_; 
v_reuseFailAlloc_1643_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1643_, 0, v_a_1637_);
v___x_1642_ = v_reuseFailAlloc_1643_;
goto v_reusejp_1641_;
}
v_reusejp_1641_:
{
return v___x_1642_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___lam__3_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2____boxed(lean_object* v___f_1645_, lean_object* v_decl_1646_, lean_object* v_stx_1647_, lean_object* v_attrKind_1648_, lean_object* v___y_1649_, lean_object* v___y_1650_, lean_object* v___y_1651_){
_start:
{
uint8_t v_attrKind_boxed_1652_; lean_object* v_res_1653_; 
v_attrKind_boxed_1652_ = lean_unbox(v_attrKind_1648_);
v_res_1653_ = lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___lam__3_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2_(v___f_1645_, v_decl_1646_, v_stx_1647_, v_attrKind_boxed_1652_, v___y_1649_, v___y_1650_);
lean_dec(v___y_1650_);
lean_dec_ref(v___y_1649_);
return v_res_1653_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___closed__19_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_1698_; lean_object* v___x_1699_; lean_object* v___x_1700_; 
v___x_1698_ = lean_unsigned_to_nat(2269118540u);
v___x_1699_ = ((lean_object*)(lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___closed__18_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2_));
v___x_1700_ = l_Lean_Name_num___override(v___x_1699_, v___x_1698_);
return v___x_1700_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___closed__21_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_1702_; lean_object* v___x_1703_; lean_object* v___x_1704_; 
v___x_1702_ = ((lean_object*)(lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___closed__20_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2_));
v___x_1703_ = lean_obj_once(&lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___closed__19_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2_, &lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___closed__19_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__once, _init_lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___closed__19_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2_);
v___x_1704_ = l_Lean_Name_str___override(v___x_1703_, v___x_1702_);
return v___x_1704_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___closed__23_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_1706_; lean_object* v___x_1707_; lean_object* v___x_1708_; 
v___x_1706_ = ((lean_object*)(lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___closed__22_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2_));
v___x_1707_ = lean_obj_once(&lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___closed__21_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2_, &lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___closed__21_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__once, _init_lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___closed__21_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2_);
v___x_1708_ = l_Lean_Name_str___override(v___x_1707_, v___x_1706_);
return v___x_1708_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___closed__24_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_1709_; lean_object* v___x_1710_; lean_object* v___x_1711_; 
v___x_1709_ = lean_unsigned_to_nat(2u);
v___x_1710_ = lean_obj_once(&lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___closed__23_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2_, &lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___closed__23_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__once, _init_lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___closed__23_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2_);
v___x_1711_ = l_Lean_Name_num___override(v___x_1710_, v___x_1709_);
return v___x_1711_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___closed__27_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2_(void){
_start:
{
uint8_t v___x_1715_; lean_object* v___x_1716_; lean_object* v___x_1717_; lean_object* v___x_1718_; lean_object* v___x_1719_; 
v___x_1715_ = 1;
v___x_1716_ = ((lean_object*)(lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___closed__26_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2_));
v___x_1717_ = ((lean_object*)(lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___closed__25_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2_));
v___x_1718_ = lean_obj_once(&lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___closed__24_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2_, &lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___closed__24_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__once, _init_lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___closed__24_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2_);
v___x_1719_ = lean_alloc_ctor(0, 3, 1);
lean_ctor_set(v___x_1719_, 0, v___x_1718_);
lean_ctor_set(v___x_1719_, 1, v___x_1717_);
lean_ctor_set(v___x_1719_, 2, v___x_1716_);
lean_ctor_set_uint8(v___x_1719_, sizeof(void*)*3, v___x_1715_);
return v___x_1719_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___closed__28_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___f_1720_; lean_object* v___f_1721_; lean_object* v___x_1722_; lean_object* v___x_1723_; 
v___f_1720_ = ((lean_object*)(lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___closed__1_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2_));
v___f_1721_ = ((lean_object*)(lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___closed__2_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2_));
v___x_1722_ = lean_obj_once(&lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___closed__27_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2_, &lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___closed__27_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__once, _init_lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___closed__27_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2_);
v___x_1723_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1723_, 0, v___x_1722_);
lean_ctor_set(v___x_1723_, 1, v___f_1721_);
lean_ctor_set(v___x_1723_, 2, v___f_1720_);
return v___x_1723_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2_(){
_start:
{
lean_object* v___x_1725_; lean_object* v___x_1726_; 
v___x_1725_ = lean_obj_once(&lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___closed__28_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2_, &lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___closed__28_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__once, _init_lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn___closed__28_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2_);
v___x_1726_ = l_Lean_registerBuiltinAttribute(v___x_1725_);
return v___x_1726_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2____boxed(lean_object* v_a_1727_){
_start:
{
lean_object* v_res_1728_; 
v_res_1728_ = lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2_();
return v_res_1728_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_ScopedEnvExtension_add___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__10(lean_object* v_00_u03b1_1729_, lean_object* v_00_u03b2_1730_, lean_object* v_00_u03c3_1731_, lean_object* v_ext_1732_, lean_object* v_b_1733_, uint8_t v_kind_1734_, lean_object* v___y_1735_, lean_object* v___y_1736_){
_start:
{
lean_object* v___x_1738_; 
v___x_1738_ = lp_aesop_Lean_ScopedEnvExtension_add___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__10___redArg(v_ext_1732_, v_b_1733_, v_kind_1734_, v___y_1735_, v___y_1736_);
return v___x_1738_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_ScopedEnvExtension_add___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__10___boxed(lean_object* v_00_u03b1_1739_, lean_object* v_00_u03b2_1740_, lean_object* v_00_u03c3_1741_, lean_object* v_ext_1742_, lean_object* v_b_1743_, lean_object* v_kind_1744_, lean_object* v___y_1745_, lean_object* v___y_1746_, lean_object* v___y_1747_){
_start:
{
uint8_t v_kind_boxed_1748_; lean_object* v_res_1749_; 
v_kind_boxed_1748_ = lean_unbox(v_kind_1744_);
v_res_1749_ = lp_aesop_Lean_ScopedEnvExtension_add___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__10(v_00_u03b1_1739_, v_00_u03b2_1740_, v_00_u03c3_1741_, v_ext_1742_, v_b_1743_, v_kind_boxed_1748_, v___y_1745_, v___y_1746_);
lean_dec(v___y_1746_);
lean_dec_ref(v___y_1745_);
return v_res_1749_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_setEnv___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__12(lean_object* v_env_1750_, lean_object* v___y_1751_, lean_object* v___y_1752_){
_start:
{
lean_object* v___x_1754_; 
v___x_1754_ = lp_aesop_Lean_setEnv___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__12___redArg(v_env_1750_, v___y_1752_);
return v___x_1754_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_setEnv___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__12___boxed(lean_object* v_env_1755_, lean_object* v___y_1756_, lean_object* v___y_1757_, lean_object* v___y_1758_){
_start:
{
lean_object* v_res_1759_; 
v_res_1759_ = lp_aesop_Lean_setEnv___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__12(v_env_1755_, v___y_1756_, v___y_1757_);
lean_dec(v___y_1757_);
lean_dec_ref(v___y_1756_);
return v_res_1759_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_getGlobalRuleSetFromData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__14(lean_object* v_ext_1760_, lean_object* v_simpExt_1761_, lean_object* v_simprocExt_1762_, lean_object* v___y_1763_, lean_object* v___y_1764_){
_start:
{
lean_object* v___x_1766_; 
v___x_1766_ = lp_aesop_Aesop_Frontend_getGlobalRuleSetFromData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__14___redArg(v_ext_1760_, v_simpExt_1761_, v_simprocExt_1762_, v___y_1764_);
return v___x_1766_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_getGlobalRuleSetFromData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__14___boxed(lean_object* v_ext_1767_, lean_object* v_simpExt_1768_, lean_object* v_simprocExt_1769_, lean_object* v___y_1770_, lean_object* v___y_1771_, lean_object* v___y_1772_){
_start:
{
lean_object* v_res_1773_; 
v_res_1773_ = lp_aesop_Aesop_Frontend_getGlobalRuleSetFromData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__14(v_ext_1767_, v_simpExt_1768_, v_simprocExt_1769_, v___y_1770_, v___y_1771_);
lean_dec(v___y_1771_);
lean_dec_ref(v___y_1770_);
lean_dec_ref(v_simprocExt_1769_);
lean_dec_ref(v_simpExt_1768_);
lean_dec_ref(v_ext_1767_);
return v_res_1773_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0_spec__0(lean_object* v_00_u03b1_1774_, lean_object* v_msg_1775_, lean_object* v___y_1776_, lean_object* v___y_1777_){
_start:
{
lean_object* v___x_1779_; 
v___x_1779_ = lp_aesop_Lean_throwError___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0_spec__0___redArg(v_msg_1775_, v___y_1776_, v___y_1777_);
return v___x_1779_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0_spec__0___boxed(lean_object* v_00_u03b1_1780_, lean_object* v_msg_1781_, lean_object* v___y_1782_, lean_object* v___y_1783_, lean_object* v___y_1784_){
_start:
{
lean_object* v_res_1785_; 
v_res_1785_ = lp_aesop_Lean_throwError___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0_spec__0(v_00_u03b1_1780_, v_msg_1781_, v___y_1782_, v___y_1783_);
lean_dec(v___y_1783_);
lean_dec_ref(v___y_1782_);
return v_res_1785_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_erase___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__11(lean_object* v_00_u03b2_1786_, lean_object* v_x_1787_, lean_object* v_x_1788_){
_start:
{
lean_object* v___x_1789_; 
v___x_1789_ = lp_aesop_Lean_PersistentHashMap_erase___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__11___redArg(v_x_1787_, v_x_1788_);
return v___x_1789_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_erase___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__11___boxed(lean_object* v_00_u03b2_1790_, lean_object* v_x_1791_, lean_object* v_x_1792_){
_start:
{
lean_object* v_res_1793_; 
v_res_1793_ = lp_aesop_Lean_PersistentHashMap_erase___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__11(v_00_u03b2_1790_, v_x_1791_, v_x_1792_);
lean_dec_ref(v_x_1792_);
return v_res_1793_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_modifyGetGlobalRuleSet___at___00__private_Aesop_Frontend_Extension_0__Aesop_Frontend_eraseGlobalRules_go___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0_spec__1_spec__3(lean_object* v_00_u03b1_1794_, lean_object* v_rsName_1795_, lean_object* v_f_1796_, lean_object* v___y_1797_, lean_object* v___y_1798_){
_start:
{
lean_object* v___x_1800_; 
v___x_1800_ = lp_aesop_Aesop_Frontend_modifyGetGlobalRuleSet___at___00__private_Aesop_Frontend_Extension_0__Aesop_Frontend_eraseGlobalRules_go___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0_spec__1_spec__3___redArg(v_rsName_1795_, v_f_1796_, v___y_1797_, v___y_1798_);
return v___x_1800_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_modifyGetGlobalRuleSet___at___00__private_Aesop_Frontend_Extension_0__Aesop_Frontend_eraseGlobalRules_go___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0_spec__1_spec__3___boxed(lean_object* v_00_u03b1_1801_, lean_object* v_rsName_1802_, lean_object* v_f_1803_, lean_object* v___y_1804_, lean_object* v___y_1805_, lean_object* v___y_1806_){
_start:
{
lean_object* v_res_1807_; 
v_res_1807_ = lp_aesop_Aesop_Frontend_modifyGetGlobalRuleSet___at___00__private_Aesop_Frontend_Extension_0__Aesop_Frontend_eraseGlobalRules_go___at___00Aesop_Frontend_eraseGlobalRules___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__0_spec__1_spec__3(v_00_u03b1_1801_, v_rsName_1802_, v_f_1803_, v___y_1804_, v___y_1805_);
lean_dec(v___y_1805_);
lean_dec_ref(v___y_1804_);
return v_res_1807_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__9_spec__12(lean_object* v_00_u03b2_1808_, lean_object* v_m_1809_, lean_object* v_a_1810_){
_start:
{
lean_object* v___x_1811_; 
v___x_1811_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__9_spec__12___redArg(v_m_1809_, v_a_1810_);
return v___x_1811_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__9_spec__12___boxed(lean_object* v_00_u03b2_1812_, lean_object* v_m_1813_, lean_object* v_a_1814_){
_start:
{
lean_object* v_res_1815_; 
v_res_1815_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__9_spec__12(v_00_u03b2_1812_, v_m_1813_, v_a_1814_);
lean_dec(v_a_1814_);
lean_dec_ref(v_m_1813_);
return v_res_1815_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_eraseAux___at___00Lean_PersistentHashMap_erase___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__11_spec__15(lean_object* v_00_u03b2_1816_, lean_object* v_x_1817_, size_t v_x_1818_, lean_object* v_x_1819_){
_start:
{
lean_object* v___x_1820_; 
v___x_1820_ = lp_aesop_Lean_PersistentHashMap_eraseAux___at___00Lean_PersistentHashMap_erase___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__11_spec__15___redArg(v_x_1817_, v_x_1818_, v_x_1819_);
return v___x_1820_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_eraseAux___at___00Lean_PersistentHashMap_erase___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__11_spec__15___boxed(lean_object* v_00_u03b2_1821_, lean_object* v_x_1822_, lean_object* v_x_1823_, lean_object* v_x_1824_){
_start:
{
size_t v_x_11873__boxed_1825_; lean_object* v_res_1826_; 
v_x_11873__boxed_1825_ = lean_unbox_usize(v_x_1823_);
lean_dec(v_x_1823_);
v_res_1826_ = lp_aesop_Lean_PersistentHashMap_eraseAux___at___00Lean_PersistentHashMap_erase___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__11_spec__15(v_00_u03b2_1821_, v_x_1822_, v_x_11873__boxed_1825_, v_x_1824_);
lean_dec_ref(v_x_1824_);
return v_res_1826_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__9_spec__12_spec__18(lean_object* v_00_u03b2_1827_, lean_object* v_a_1828_, lean_object* v_x_1829_){
_start:
{
lean_object* v___x_1830_; 
v___x_1830_ = lp_aesop_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__9_spec__12_spec__18___redArg(v_a_1828_, v_x_1829_);
return v___x_1830_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__9_spec__12_spec__18___boxed(lean_object* v_00_u03b2_1831_, lean_object* v_a_1832_, lean_object* v_x_1833_){
_start:
{
lean_object* v_res_1834_; 
v_res_1834_ = lp_aesop_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00Aesop_Frontend_getGlobalRuleSetData___at___00Aesop_Frontend_addGlobalRule___at___00__private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2__spec__2_spec__9_spec__12_spec__18(v_00_u03b2_1831_, v_a_1832_, v_x_1833_);
lean_dec(v_x_1833_);
lean_dec(v_a_1832_);
return v_res_1834_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
void lean_initialize_runtime_module();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_aesop_Aesop_Frontend_Attribute(uint8_t builtin) {
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
lean_object* runtime_initialize_aesop_Aesop_Frontend_Extension(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_Frontend_RuleExpr(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_aesop_Aesop_Frontend_Attribute(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Frontend_Extension(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Frontend_RuleExpr(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_aesop_Lean_Parser_Category_Aesop_attr__rules = _init_lp_aesop_Lean_Parser_Category_Aesop_attr__rules();
lean_mark_persistent(lp_aesop_Lean_Parser_Category_Aesop_attr__rules);
res = lp_aesop___private_Aesop_Frontend_Attribute_0__Aesop_Frontend_initFn_00___x40_Aesop_Frontend_Attribute_2269118540____hygCtx___hyg_2_();
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_aesop_Aesop_Frontend_Extension(uint8_t builtin);
lean_object* initialize_aesop_Aesop_Frontend_RuleExpr(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_aesop_Aesop_Frontend_Attribute(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Frontend_Extension(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Frontend_RuleExpr(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Frontend_Attribute(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_aesop_Aesop_Frontend_Attribute(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_aesop_Aesop_Frontend_Attribute(builtin);
}
#ifdef __cplusplus
}
#endif
