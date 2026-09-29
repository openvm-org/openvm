// Lean compiler output
// Module: Aesop.Frontend.Saturate
// Imports: public import Init public meta import Init public meta import Aesop.Saturate public meta import Aesop.Frontend.Extension public meta import Aesop.Builder.Forward public meta import Aesop.Stats.File
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
lean_object* l_Lean_Meta_ConfigWithKey_setTransparency(uint8_t, lean_object*);
lean_object* l_Lean_FVarId_getType___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_get_size(lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* l_Lean_Meta_isProp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_forallTelescopeReducingAux(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_fvar___override(lean_object*);
lean_object* l_Lean_PrettyPrinter_delab(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* lp_aesop_Aesop_RuleBuilderOptions_default;
lean_object* lean_nat_to_int(lean_object*);
lean_object* lp_aesop_Aesop_RuleBuilder_forward(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr3(lean_object*, lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lp_aesop_Aesop_getDefaultRuleSetNames();
size_t lean_array_size(lean_object*);
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* lean_array_uget(lean_object*, size_t);
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
lean_object* l_Lean_TSyntax_getId(lean_object*);
size_t lean_usize_add(size_t, size_t);
lean_object* l_Array_append___redArg(lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_Frontend_getGlobalRuleSets(lean_object*, lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
size_t lean_usize_of_nat(lean_object*);
uint8_t lean_usize_dec_eq(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* lean_io_error_to_string(lean_object*);
lean_object* l_Lean_MessageData_ofFormat(lean_object*);
lean_object* lp_aesop_Aesop_mkLocalRuleSet(lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
extern lean_object* l_Lean_Elab_unsupportedSyntaxExceptionId;
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_LocalRuleSet_add(lean_object*, lean_object*);
lean_object* l_Lean_MVarId_getDecl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_LocalDecl_isImplementationDetail(lean_object*);
lean_object* l_Lean_LocalDecl_fvarId(lean_object*);
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_withMVarContextImp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArgs(lean_object*);
lean_object* l_Lean_Syntax_TSepArray_getElems___redArg(lean_object*);
extern lean_object* lp_aesop_Aesop_aesop_dev_generateScript;
lean_object* l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(lean_object*, lean_object*);
extern lean_object* lp_aesop_Aesop_Check_script;
uint8_t lp_aesop_Aesop_Check_get(lean_object*, lean_object*);
extern lean_object* lp_aesop_Aesop_Check_script_steps;
lean_object* l_Lean_Elab_Tactic_getMainGoal___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_saturate(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_replaceMainGoal___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_withMainContext___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* lp_aesop_Aesop_aesop_stats_file;
uint8_t lean_string_dec_eq(lean_object*, lean_object*);
lean_object* l_Lean_Elab_Term_getDeclName_x3f___redArg(lean_object*);
lean_object* l_Lean_PrettyPrinter_ppCategory(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Std_Format_pretty(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getPos_x3f(lean_object*, uint8_t);
lean_object* l_Lean_FileMap_toPosition(lean_object*, lean_object*);
lean_object* lean_io_prim_handle_mk(lean_object*, uint8_t);
lean_object* lean_io_prim_handle_lock(lean_object*, uint8_t);
lean_object* lp_aesop_Aesop_instToJsonStatsFileRecord_toJson(lean_object*);
lean_object* l_Lean_Json_compress(lean_object*);
lean_object* l_IO_FS_Handle_putStrLn(lean_object*, lean_object*);
lean_object* lean_io_prim_handle_unlock(lean_object*);
lean_object* l_Lean_TSyntax_getNat(lean_object*);
uint8_t l_Lean_Syntax_isNone(lean_object*);
uint8_t l_Lean_Syntax_matchesNull(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Array_mkArray1___redArg(lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Syntax_node1(lean_object*, lean_object*, lean_object*);
lean_object* l_Array_mkArray0(lean_object*);
lean_object* l_Lean_Syntax_getOptional_x3f(lean_object*);
static const lean_string_object lp_aesop_Aesop_Frontend_Parser_usingRuleSets___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "usingRuleSets"};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_usingRuleSets___closed__0 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_usingRuleSets___closed__0_value;
static const lean_string_object lp_aesop_Aesop_Frontend_Parser_usingRuleSets___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "Aesop"};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_usingRuleSets___closed__1 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_usingRuleSets___closed__1_value;
static const lean_string_object lp_aesop_Aesop_Frontend_Parser_usingRuleSets___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "Frontend"};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_usingRuleSets___closed__2 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_usingRuleSets___closed__2_value;
static const lean_string_object lp_aesop_Aesop_Frontend_Parser_usingRuleSets___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_usingRuleSets___closed__3 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_usingRuleSets___closed__3_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_usingRuleSets___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_usingRuleSets___closed__1_value),LEAN_SCALAR_PTR_LITERAL(97, 226, 101, 135, 78, 117, 164, 248)}};
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_usingRuleSets___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_usingRuleSets___closed__4_value_aux_0),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_usingRuleSets___closed__2_value),LEAN_SCALAR_PTR_LITERAL(212, 61, 86, 210, 90, 71, 197, 0)}};
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_usingRuleSets___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_usingRuleSets___closed__4_value_aux_1),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_usingRuleSets___closed__3_value),LEAN_SCALAR_PTR_LITERAL(13, 69, 189, 207, 58, 198, 186, 160)}};
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_usingRuleSets___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_usingRuleSets___closed__4_value_aux_2),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_usingRuleSets___closed__0_value),LEAN_SCALAR_PTR_LITERAL(140, 176, 190, 218, 212, 148, 225, 108)}};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_usingRuleSets___closed__4 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_usingRuleSets___closed__4_value;
static const lean_string_object lp_aesop_Aesop_Frontend_Parser_usingRuleSets___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_usingRuleSets___closed__5 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_usingRuleSets___closed__5_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_usingRuleSets___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_usingRuleSets___closed__5_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_usingRuleSets___closed__6 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_usingRuleSets___closed__6_value;
static const lean_string_object lp_aesop_Aesop_Frontend_Parser_usingRuleSets___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "using "};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_usingRuleSets___closed__7 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_usingRuleSets___closed__7_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_usingRuleSets___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_usingRuleSets___closed__7_value)}};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_usingRuleSets___closed__8 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_usingRuleSets___closed__8_value;
static const lean_string_object lp_aesop_Aesop_Frontend_Parser_usingRuleSets___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "many1"};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_usingRuleSets___closed__9 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_usingRuleSets___closed__9_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_usingRuleSets___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_usingRuleSets___closed__9_value),LEAN_SCALAR_PTR_LITERAL(55, 136, 52, 6, 12, 19, 78, 239)}};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_usingRuleSets___closed__10 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_usingRuleSets___closed__10_value;
static const lean_string_object lp_aesop_Aesop_Frontend_Parser_usingRuleSets___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ident"};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_usingRuleSets___closed__11 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_usingRuleSets___closed__11_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_usingRuleSets___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_usingRuleSets___closed__11_value),LEAN_SCALAR_PTR_LITERAL(52, 159, 208, 51, 14, 60, 6, 71)}};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_usingRuleSets___closed__12 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_usingRuleSets___closed__12_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_usingRuleSets___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_usingRuleSets___closed__12_value)}};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_usingRuleSets___closed__13 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_usingRuleSets___closed__13_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_usingRuleSets___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_usingRuleSets___closed__10_value),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_usingRuleSets___closed__13_value)}};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_usingRuleSets___closed__14 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_usingRuleSets___closed__14_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_usingRuleSets___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_usingRuleSets___closed__6_value),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_usingRuleSets___closed__8_value),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_usingRuleSets___closed__14_value)}};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_usingRuleSets___closed__15 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_usingRuleSets___closed__15_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_usingRuleSets___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 9}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_usingRuleSets___closed__0_value),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_usingRuleSets___closed__4_value),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_usingRuleSets___closed__15_value)}};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_usingRuleSets___closed__16 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_usingRuleSets___closed__16_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_Frontend_Parser_usingRuleSets = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_usingRuleSets___closed__16_value;
static const lean_string_object lp_aesop_Aesop_Frontend_Parser_additionalRule___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "additionalRule"};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_additionalRule___closed__0 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_additionalRule___closed__0_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_additionalRule___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_usingRuleSets___closed__1_value),LEAN_SCALAR_PTR_LITERAL(97, 226, 101, 135, 78, 117, 164, 248)}};
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_additionalRule___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_additionalRule___closed__1_value_aux_0),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_usingRuleSets___closed__2_value),LEAN_SCALAR_PTR_LITERAL(212, 61, 86, 210, 90, 71, 197, 0)}};
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_additionalRule___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_additionalRule___closed__1_value_aux_1),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_usingRuleSets___closed__3_value),LEAN_SCALAR_PTR_LITERAL(13, 69, 189, 207, 58, 198, 186, 160)}};
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_additionalRule___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_additionalRule___closed__1_value_aux_2),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_additionalRule___closed__0_value),LEAN_SCALAR_PTR_LITERAL(52, 174, 125, 142, 139, 101, 194, 50)}};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_additionalRule___closed__1 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_additionalRule___closed__1_value;
static const lean_string_object lp_aesop_Aesop_Frontend_Parser_additionalRule___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "orelse"};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_additionalRule___closed__2 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_additionalRule___closed__2_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_additionalRule___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_additionalRule___closed__2_value),LEAN_SCALAR_PTR_LITERAL(78, 76, 4, 51, 251, 212, 116, 5)}};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_additionalRule___closed__3 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_additionalRule___closed__3_value;
static const lean_string_object lp_aesop_Aesop_Frontend_Parser_additionalRule___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "*"};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_additionalRule___closed__4 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_additionalRule___closed__4_value;
static const lean_string_object lp_aesop_Aesop_Frontend_Parser_additionalRule___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "token"};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_additionalRule___closed__5 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_additionalRule___closed__5_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_additionalRule___closed__6_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_additionalRule___closed__5_value),LEAN_SCALAR_PTR_LITERAL(89, 149, 26, 37, 31, 104, 89, 130)}};
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_additionalRule___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_additionalRule___closed__6_value_aux_0),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_additionalRule___closed__4_value),LEAN_SCALAR_PTR_LITERAL(46, 123, 149, 63, 0, 221, 179, 78)}};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_additionalRule___closed__6 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_additionalRule___closed__6_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_additionalRule___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_additionalRule___closed__4_value)}};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_additionalRule___closed__7 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_additionalRule___closed__7_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_additionalRule___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 9}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_additionalRule___closed__4_value),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_additionalRule___closed__6_value),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_additionalRule___closed__7_value)}};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_additionalRule___closed__8 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_additionalRule___closed__8_value;
static const lean_string_object lp_aesop_Aesop_Frontend_Parser_additionalRule___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "term"};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_additionalRule___closed__9 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_additionalRule___closed__9_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_additionalRule___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_additionalRule___closed__9_value),LEAN_SCALAR_PTR_LITERAL(187, 230, 181, 162, 253, 146, 122, 119)}};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_additionalRule___closed__10 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_additionalRule___closed__10_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_additionalRule___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_additionalRule___closed__10_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_additionalRule___closed__11 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_additionalRule___closed__11_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_additionalRule___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_additionalRule___closed__3_value),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_additionalRule___closed__8_value),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_additionalRule___closed__11_value)}};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_additionalRule___closed__12 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_additionalRule___closed__12_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_additionalRule___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 9}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_additionalRule___closed__0_value),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_additionalRule___closed__1_value),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_additionalRule___closed__12_value)}};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_additionalRule___closed__13 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_additionalRule___closed__13_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_Frontend_Parser_additionalRule = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_additionalRule___closed__13_value;
static const lean_string_object lp_aesop_Aesop_Frontend_Parser_additionalRules___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "additionalRules"};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_additionalRules___closed__0 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_additionalRules___closed__0_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_additionalRules___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_usingRuleSets___closed__1_value),LEAN_SCALAR_PTR_LITERAL(97, 226, 101, 135, 78, 117, 164, 248)}};
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_additionalRules___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_additionalRules___closed__1_value_aux_0),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_usingRuleSets___closed__2_value),LEAN_SCALAR_PTR_LITERAL(212, 61, 86, 210, 90, 71, 197, 0)}};
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_additionalRules___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_additionalRules___closed__1_value_aux_1),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_usingRuleSets___closed__3_value),LEAN_SCALAR_PTR_LITERAL(13, 69, 189, 207, 58, 198, 186, 160)}};
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_additionalRules___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_additionalRules___closed__1_value_aux_2),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_additionalRules___closed__0_value),LEAN_SCALAR_PTR_LITERAL(170, 13, 39, 193, 9, 166, 96, 39)}};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_additionalRules___closed__1 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_additionalRules___closed__1_value;
static const lean_string_object lp_aesop_Aesop_Frontend_Parser_additionalRules___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "["};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_additionalRules___closed__2 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_additionalRules___closed__2_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_additionalRules___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_additionalRules___closed__2_value)}};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_additionalRules___closed__3 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_additionalRules___closed__3_value;
static const lean_string_object lp_aesop_Aesop_Frontend_Parser_additionalRules___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ","};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_additionalRules___closed__4 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_additionalRules___closed__4_value;
static const lean_string_object lp_aesop_Aesop_Frontend_Parser_additionalRules___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = ", "};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_additionalRules___closed__5 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_additionalRules___closed__5_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_additionalRules___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_additionalRules___closed__5_value)}};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_additionalRules___closed__6 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_additionalRules___closed__6_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_additionalRules___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 8, .m_other = 3, .m_tag = 10}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_additionalRule___closed__13_value),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_additionalRules___closed__4_value),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_additionalRules___closed__6_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_additionalRules___closed__7 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_additionalRules___closed__7_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_additionalRules___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_usingRuleSets___closed__6_value),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_additionalRules___closed__3_value),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_additionalRules___closed__7_value)}};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_additionalRules___closed__8 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_additionalRules___closed__8_value;
static const lean_string_object lp_aesop_Aesop_Frontend_Parser_additionalRules___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "]"};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_additionalRules___closed__9 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_additionalRules___closed__9_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_additionalRules___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_additionalRules___closed__9_value)}};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_additionalRules___closed__10 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_additionalRules___closed__10_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_additionalRules___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_usingRuleSets___closed__6_value),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_additionalRules___closed__8_value),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_additionalRules___closed__10_value)}};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_additionalRules___closed__11 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_additionalRules___closed__11_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_Parser_additionalRules___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 9}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_additionalRules___closed__0_value),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_additionalRules___closed__1_value),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_additionalRules___closed__11_value)}};
static const lean_object* lp_aesop_Aesop_Frontend_Parser_additionalRules___closed__12 = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_additionalRules___closed__12_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_Frontend_Parser_additionalRules = (const lean_object*)&lp_aesop_Aesop_Frontend_Parser_additionalRules___closed__12_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_ElabM_runForwardElab___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ElabM_runForwardElab___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ElabM_runForwardElab(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ElabM_runForwardElab___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Check_isEnabled___at___00Aesop_Options_toOptions_x27___at___00Aesop_Frontend_mkForwardOptions_spec__0_spec__1___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Check_isEnabled___at___00Aesop_Options_toOptions_x27___at___00Aesop_Frontend_mkForwardOptions_spec__0_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Lean_Option_get___at___00Aesop_Options_toOptions_x27___at___00Aesop_Frontend_mkForwardOptions_spec__0_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_get___at___00Aesop_Options_toOptions_x27___at___00Aesop_Frontend_mkForwardOptions_spec__0_spec__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Options_toOptions_x27___at___00Aesop_Frontend_mkForwardOptions_spec__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Options_toOptions_x27___at___00Aesop_Frontend_mkForwardOptions_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_mkForwardOptions(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_mkForwardOptions___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Check_isEnabled___at___00Aesop_Options_toOptions_x27___at___00Aesop_Frontend_mkForwardOptions_spec__0_spec__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Check_isEnabled___at___00Aesop_Options_toOptions_x27___at___00Aesop_Frontend_mkForwardOptions_spec__0_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00Aesop_Frontend_elabGlobalRuleSets_spec__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Frontend_elabGlobalRuleSets_spec__2(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Frontend_elabGlobalRuleSets_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Frontend_elabGlobalRuleSets_spec__0(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Frontend_elabGlobalRuleSets_spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_elabGlobalRuleSets(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_elabGlobalRuleSets___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_aesop_Aesop_Frontend_elabForwardRule___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Frontend_elabForwardRule___closed__0;
static lean_once_cell_t lp_aesop_Aesop_Frontend_elabForwardRule___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Frontend_elabForwardRule___closed__1;
static lean_once_cell_t lp_aesop_Aesop_Frontend_elabForwardRule___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Frontend_elabForwardRule___closed__2;
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_elabForwardRule(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_elabForwardRule___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_mkHypForwardRule(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_mkHypForwardRule___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_forallBoundedTelescope___at___00Aesop_Frontend_isImplication_spec__0___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_forallBoundedTelescope___at___00Aesop_Frontend_isImplication_spec__0___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_forallBoundedTelescope___at___00Aesop_Frontend_isImplication_spec__0___redArg(lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_forallBoundedTelescope___at___00Aesop_Frontend_isImplication_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_forallBoundedTelescope___at___00Aesop_Frontend_isImplication_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_forallBoundedTelescope___at___00Aesop_Frontend_isImplication_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_isImplication___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_isImplication___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_aesop_Aesop_Frontend_isImplication___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_Frontend_isImplication___lam__0___boxed, .m_arity = 7, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_Frontend_isImplication___closed__0 = (const lean_object*)&lp_aesop_Aesop_Frontend_isImplication___closed__0_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_isImplication___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(1) << 1) | 1))}};
static const lean_object* lp_aesop_Aesop_Frontend_isImplication___closed__1 = (const lean_object*)&lp_aesop_Aesop_Frontend_isImplication___closed__1_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_isImplication(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_isImplication___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_Frontend_mkHypImplicationRule_x3f_spec__0___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_Frontend_mkHypImplicationRule_x3f_spec__0___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_Frontend_mkHypImplicationRule_x3f_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_Frontend_mkHypImplicationRule_x3f_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_Frontend_mkHypImplicationRule_x3f_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_Frontend_mkHypImplicationRule_x3f_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_mkHypImplicationRule_x3f___lam__0(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_mkHypImplicationRule_x3f___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_mkHypImplicationRule_x3f(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_mkHypImplicationRule_x3f___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_Frontend_addLocalImplications_spec__0_spec__0_spec__2_spec__3(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_Frontend_addLocalImplications_spec__0_spec__0_spec__2_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_Frontend_addLocalImplications_spec__0_spec__0_spec__2(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_Frontend_addLocalImplications_spec__0_spec__0_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_Frontend_addLocalImplications_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_Frontend_addLocalImplications_spec__0_spec__0_spec__1(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_Frontend_addLocalImplications_spec__0_spec__0_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_Frontend_addLocalImplications_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Aesop_Frontend_addLocalImplications_spec__0_spec__1_spec__4(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Aesop_Frontend_addLocalImplications_spec__0_spec__1_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Aesop_Frontend_addLocalImplications_spec__0_spec__1(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Aesop_Frontend_addLocalImplications_spec__0_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentArray_forIn___at___00Aesop_Frontend_addLocalImplications_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentArray_forIn___at___00Aesop_Frontend_addLocalImplications_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_addLocalImplications(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_addLocalImplications___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_aesop_Lean_Elab_throwUnsupportedSyntax___at___00Aesop_Frontend_elabAdditionalForwardRules_spec__0___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Lean_Elab_throwUnsupportedSyntax___at___00Aesop_Frontend_elabAdditionalForwardRules_spec__0___redArg___closed__0;
LEAN_EXPORT lean_object* lp_aesop_Lean_Elab_throwUnsupportedSyntax___at___00Aesop_Frontend_elabAdditionalForwardRules_spec__0___redArg();
LEAN_EXPORT lean_object* lp_aesop_Lean_Elab_throwUnsupportedSyntax___at___00Aesop_Frontend_elabAdditionalForwardRules_spec__0___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Elab_throwUnsupportedSyntax___at___00Aesop_Frontend_elabAdditionalForwardRules_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Elab_throwUnsupportedSyntax___at___00Aesop_Frontend_elabAdditionalForwardRules_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Frontend_elabAdditionalForwardRules_spec__1(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Frontend_elabAdditionalForwardRules_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_elabAdditionalForwardRules(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_elabAdditionalForwardRules___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_elabForwardRuleSetCore(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_elabForwardRuleSetCore___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_aesop_Aesop_Frontend_elabForwardRuleSet___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_aesop_Aesop_Frontend_elabForwardRuleSet___closed__0 = (const lean_object*)&lp_aesop_Aesop_Frontend_elabForwardRuleSet___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_elabForwardRuleSet(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_elabForwardRuleSet___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_evalSaturateCore___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_evalSaturateCore___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_get___at___00Aesop_appendStatsToStatsFileIfEnabled___at___00Aesop_Frontend_evalSaturateCore_spec__0_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_get___at___00Aesop_appendStatsToStatsFileIfEnabled___at___00Aesop_Frontend_evalSaturateCore_spec__0_spec__0___boxed(lean_object*, lean_object*);
static const lean_string_object lp_aesop_Aesop_StatsFileRecord_ofStats___at___00Aesop_appendStatsToStatsFileIfEnabled___at___00Aesop_Frontend_evalSaturateCore_spec__0_spec__1___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "tactic"};
static const lean_object* lp_aesop_Aesop_StatsFileRecord_ofStats___at___00Aesop_appendStatsToStatsFileIfEnabled___at___00Aesop_Frontend_evalSaturateCore_spec__0_spec__1___redArg___closed__0 = (const lean_object*)&lp_aesop_Aesop_StatsFileRecord_ofStats___at___00Aesop_appendStatsToStatsFileIfEnabled___at___00Aesop_Frontend_evalSaturateCore_spec__0_spec__1___redArg___closed__0_value;
static const lean_ctor_object lp_aesop_Aesop_StatsFileRecord_ofStats___at___00Aesop_appendStatsToStatsFileIfEnabled___at___00Aesop_Frontend_evalSaturateCore_spec__0_spec__1___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_StatsFileRecord_ofStats___at___00Aesop_appendStatsToStatsFileIfEnabled___at___00Aesop_Frontend_evalSaturateCore_spec__0_spec__1___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(99, 76, 33, 121, 85, 143, 17, 224)}};
static const lean_object* lp_aesop_Aesop_StatsFileRecord_ofStats___at___00Aesop_appendStatsToStatsFileIfEnabled___at___00Aesop_Frontend_evalSaturateCore_spec__0_spec__1___redArg___closed__1 = (const lean_object*)&lp_aesop_Aesop_StatsFileRecord_ofStats___at___00Aesop_appendStatsToStatsFileIfEnabled___at___00Aesop_Frontend_evalSaturateCore_spec__0_spec__1___redArg___closed__1_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_StatsFileRecord_ofStats___at___00Aesop_appendStatsToStatsFileIfEnabled___at___00Aesop_Frontend_evalSaturateCore_spec__0_spec__1___redArg(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_StatsFileRecord_ofStats___at___00Aesop_appendStatsToStatsFileIfEnabled___at___00Aesop_Frontend_evalSaturateCore_spec__0_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop_Aesop_appendStatsToStatsFileIfEnabled___at___00Aesop_Frontend_evalSaturateCore_spec__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1, .m_capacity = 1, .m_length = 0, .m_data = ""};
static const lean_object* lp_aesop_Aesop_appendStatsToStatsFileIfEnabled___at___00Aesop_Frontend_evalSaturateCore_spec__0___closed__0 = (const lean_object*)&lp_aesop_Aesop_appendStatsToStatsFileIfEnabled___at___00Aesop_Frontend_evalSaturateCore_spec__0___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_appendStatsToStatsFileIfEnabled___at___00Aesop_Frontend_evalSaturateCore_spec__0(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_appendStatsToStatsFileIfEnabled___at___00Aesop_Frontend_evalSaturateCore_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_evalSaturateCore(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_evalSaturateCore___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_StatsFileRecord_ofStats___at___00Aesop_appendStatsToStatsFileIfEnabled___at___00Aesop_Frontend_evalSaturateCore_spec__0_spec__1(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_StatsFileRecord_ofStats___at___00Aesop_appendStatsToStatsFileIfEnabled___at___00Aesop_Frontend_evalSaturateCore_spec__0_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop_Aesop_Frontend_saturate___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "saturate"};
static const lean_object* lp_aesop_Aesop_Frontend_saturate___closed__0 = (const lean_object*)&lp_aesop_Aesop_Frontend_saturate___closed__0_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_saturate___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_usingRuleSets___closed__1_value),LEAN_SCALAR_PTR_LITERAL(97, 226, 101, 135, 78, 117, 164, 248)}};
static const lean_ctor_object lp_aesop_Aesop_Frontend_saturate___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_saturate___closed__1_value_aux_0),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_usingRuleSets___closed__2_value),LEAN_SCALAR_PTR_LITERAL(212, 61, 86, 210, 90, 71, 197, 0)}};
static const lean_ctor_object lp_aesop_Aesop_Frontend_saturate___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_saturate___closed__1_value_aux_1),((lean_object*)&lp_aesop_Aesop_Frontend_saturate___closed__0_value),LEAN_SCALAR_PTR_LITERAL(5, 209, 111, 93, 249, 255, 235, 47)}};
static const lean_object* lp_aesop_Aesop_Frontend_saturate___closed__1 = (const lean_object*)&lp_aesop_Aesop_Frontend_saturate___closed__1_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_saturate___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_saturate___closed__0_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_aesop_Aesop_Frontend_saturate___closed__2 = (const lean_object*)&lp_aesop_Aesop_Frontend_saturate___closed__2_value;
static const lean_string_object lp_aesop_Aesop_Frontend_saturate___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "optional"};
static const lean_object* lp_aesop_Aesop_Frontend_saturate___closed__3 = (const lean_object*)&lp_aesop_Aesop_Frontend_saturate___closed__3_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_saturate___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_Frontend_saturate___closed__3_value),LEAN_SCALAR_PTR_LITERAL(233, 141, 154, 50, 143, 135, 42, 252)}};
static const lean_object* lp_aesop_Aesop_Frontend_saturate___closed__4 = (const lean_object*)&lp_aesop_Aesop_Frontend_saturate___closed__4_value;
static const lean_string_object lp_aesop_Aesop_Frontend_saturate___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "ppSpace"};
static const lean_object* lp_aesop_Aesop_Frontend_saturate___closed__5 = (const lean_object*)&lp_aesop_Aesop_Frontend_saturate___closed__5_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_saturate___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_Frontend_saturate___closed__5_value),LEAN_SCALAR_PTR_LITERAL(207, 47, 58, 43, 30, 240, 125, 246)}};
static const lean_object* lp_aesop_Aesop_Frontend_saturate___closed__6 = (const lean_object*)&lp_aesop_Aesop_Frontend_saturate___closed__6_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_saturate___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_saturate___closed__6_value)}};
static const lean_object* lp_aesop_Aesop_Frontend_saturate___closed__7 = (const lean_object*)&lp_aesop_Aesop_Frontend_saturate___closed__7_value;
static const lean_string_object lp_aesop_Aesop_Frontend_saturate___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "num"};
static const lean_object* lp_aesop_Aesop_Frontend_saturate___closed__8 = (const lean_object*)&lp_aesop_Aesop_Frontend_saturate___closed__8_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_saturate___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_Frontend_saturate___closed__8_value),LEAN_SCALAR_PTR_LITERAL(227, 68, 22, 222, 47, 51, 204, 84)}};
static const lean_object* lp_aesop_Aesop_Frontend_saturate___closed__9 = (const lean_object*)&lp_aesop_Aesop_Frontend_saturate___closed__9_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_saturate___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_saturate___closed__9_value)}};
static const lean_object* lp_aesop_Aesop_Frontend_saturate___closed__10 = (const lean_object*)&lp_aesop_Aesop_Frontend_saturate___closed__10_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_saturate___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_usingRuleSets___closed__6_value),((lean_object*)&lp_aesop_Aesop_Frontend_saturate___closed__7_value),((lean_object*)&lp_aesop_Aesop_Frontend_saturate___closed__10_value)}};
static const lean_object* lp_aesop_Aesop_Frontend_saturate___closed__11 = (const lean_object*)&lp_aesop_Aesop_Frontend_saturate___closed__11_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_saturate___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_saturate___closed__4_value),((lean_object*)&lp_aesop_Aesop_Frontend_saturate___closed__11_value)}};
static const lean_object* lp_aesop_Aesop_Frontend_saturate___closed__12 = (const lean_object*)&lp_aesop_Aesop_Frontend_saturate___closed__12_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_saturate___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_usingRuleSets___closed__6_value),((lean_object*)&lp_aesop_Aesop_Frontend_saturate___closed__2_value),((lean_object*)&lp_aesop_Aesop_Frontend_saturate___closed__12_value)}};
static const lean_object* lp_aesop_Aesop_Frontend_saturate___closed__13 = (const lean_object*)&lp_aesop_Aesop_Frontend_saturate___closed__13_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_saturate___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_usingRuleSets___closed__6_value),((lean_object*)&lp_aesop_Aesop_Frontend_saturate___closed__7_value),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_additionalRules___closed__12_value)}};
static const lean_object* lp_aesop_Aesop_Frontend_saturate___closed__14 = (const lean_object*)&lp_aesop_Aesop_Frontend_saturate___closed__14_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_saturate___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_saturate___closed__4_value),((lean_object*)&lp_aesop_Aesop_Frontend_saturate___closed__14_value)}};
static const lean_object* lp_aesop_Aesop_Frontend_saturate___closed__15 = (const lean_object*)&lp_aesop_Aesop_Frontend_saturate___closed__15_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_saturate___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_usingRuleSets___closed__6_value),((lean_object*)&lp_aesop_Aesop_Frontend_saturate___closed__13_value),((lean_object*)&lp_aesop_Aesop_Frontend_saturate___closed__15_value)}};
static const lean_object* lp_aesop_Aesop_Frontend_saturate___closed__16 = (const lean_object*)&lp_aesop_Aesop_Frontend_saturate___closed__16_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_saturate___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_usingRuleSets___closed__6_value),((lean_object*)&lp_aesop_Aesop_Frontend_saturate___closed__7_value),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_usingRuleSets___closed__16_value)}};
static const lean_object* lp_aesop_Aesop_Frontend_saturate___closed__17 = (const lean_object*)&lp_aesop_Aesop_Frontend_saturate___closed__17_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_saturate___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_saturate___closed__4_value),((lean_object*)&lp_aesop_Aesop_Frontend_saturate___closed__17_value)}};
static const lean_object* lp_aesop_Aesop_Frontend_saturate___closed__18 = (const lean_object*)&lp_aesop_Aesop_Frontend_saturate___closed__18_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_saturate___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_usingRuleSets___closed__6_value),((lean_object*)&lp_aesop_Aesop_Frontend_saturate___closed__16_value),((lean_object*)&lp_aesop_Aesop_Frontend_saturate___closed__18_value)}};
static const lean_object* lp_aesop_Aesop_Frontend_saturate___closed__19 = (const lean_object*)&lp_aesop_Aesop_Frontend_saturate___closed__19_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_saturate___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_saturate___closed__1_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_Frontend_saturate___closed__19_value)}};
static const lean_object* lp_aesop_Aesop_Frontend_saturate___closed__20 = (const lean_object*)&lp_aesop_Aesop_Frontend_saturate___closed__20_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_Frontend_saturate = (const lean_object*)&lp_aesop_Aesop_Frontend_saturate___closed__20_value;
static const lean_string_object lp_aesop_Aesop_Frontend_saturate_x3f___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "saturate\?"};
static const lean_object* lp_aesop_Aesop_Frontend_saturate_x3f___closed__0 = (const lean_object*)&lp_aesop_Aesop_Frontend_saturate_x3f___closed__0_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_saturate_x3f___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_usingRuleSets___closed__1_value),LEAN_SCALAR_PTR_LITERAL(97, 226, 101, 135, 78, 117, 164, 248)}};
static const lean_ctor_object lp_aesop_Aesop_Frontend_saturate_x3f___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_saturate_x3f___closed__1_value_aux_0),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_usingRuleSets___closed__2_value),LEAN_SCALAR_PTR_LITERAL(212, 61, 86, 210, 90, 71, 197, 0)}};
static const lean_ctor_object lp_aesop_Aesop_Frontend_saturate_x3f___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_saturate_x3f___closed__1_value_aux_1),((lean_object*)&lp_aesop_Aesop_Frontend_saturate_x3f___closed__0_value),LEAN_SCALAR_PTR_LITERAL(96, 66, 159, 112, 146, 5, 116, 238)}};
static const lean_object* lp_aesop_Aesop_Frontend_saturate_x3f___closed__1 = (const lean_object*)&lp_aesop_Aesop_Frontend_saturate_x3f___closed__1_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_saturate_x3f___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_saturate_x3f___closed__0_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_aesop_Aesop_Frontend_saturate_x3f___closed__2 = (const lean_object*)&lp_aesop_Aesop_Frontend_saturate_x3f___closed__2_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_saturate_x3f___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_usingRuleSets___closed__6_value),((lean_object*)&lp_aesop_Aesop_Frontend_saturate_x3f___closed__2_value),((lean_object*)&lp_aesop_Aesop_Frontend_saturate___closed__12_value)}};
static const lean_object* lp_aesop_Aesop_Frontend_saturate_x3f___closed__3 = (const lean_object*)&lp_aesop_Aesop_Frontend_saturate_x3f___closed__3_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_saturate_x3f___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_usingRuleSets___closed__6_value),((lean_object*)&lp_aesop_Aesop_Frontend_saturate_x3f___closed__3_value),((lean_object*)&lp_aesop_Aesop_Frontend_saturate___closed__15_value)}};
static const lean_object* lp_aesop_Aesop_Frontend_saturate_x3f___closed__4 = (const lean_object*)&lp_aesop_Aesop_Frontend_saturate_x3f___closed__4_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_saturate_x3f___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_usingRuleSets___closed__6_value),((lean_object*)&lp_aesop_Aesop_Frontend_saturate_x3f___closed__4_value),((lean_object*)&lp_aesop_Aesop_Frontend_saturate___closed__18_value)}};
static const lean_object* lp_aesop_Aesop_Frontend_saturate_x3f___closed__5 = (const lean_object*)&lp_aesop_Aesop_Frontend_saturate_x3f___closed__5_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_saturate_x3f___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_saturate_x3f___closed__1_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_Frontend_saturate_x3f___closed__5_value)}};
static const lean_object* lp_aesop_Aesop_Frontend_saturate_x3f___closed__6 = (const lean_object*)&lp_aesop_Aesop_Frontend_saturate_x3f___closed__6_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_Frontend_saturate_x3f = (const lean_object*)&lp_aesop_Aesop_Frontend_saturate_x3f___closed__6_value;
LEAN_EXPORT lean_object* lp_aesop_Lean_Elab_throwUnsupportedSyntax___at___00Aesop_Frontend_evalSaturate_spec__0___redArg();
LEAN_EXPORT lean_object* lp_aesop_Lean_Elab_throwUnsupportedSyntax___at___00Aesop_Frontend_evalSaturate_spec__0___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Elab_throwUnsupportedSyntax___at___00Aesop_Frontend_evalSaturate_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Elab_throwUnsupportedSyntax___at___00Aesop_Frontend_evalSaturate_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_evalSaturate(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_evalSaturate___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_evalSaturate_x3f(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_evalSaturate_x3f___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop_Aesop_Frontend_tacticForward_________00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "tacticForward____"};
static const lean_object* lp_aesop_Aesop_Frontend_tacticForward_________00__closed__0 = (const lean_object*)&lp_aesop_Aesop_Frontend_tacticForward_________00__closed__0_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_tacticForward_________00__closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_usingRuleSets___closed__1_value),LEAN_SCALAR_PTR_LITERAL(97, 226, 101, 135, 78, 117, 164, 248)}};
static const lean_ctor_object lp_aesop_Aesop_Frontend_tacticForward_________00__closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_tacticForward_________00__closed__1_value_aux_0),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_usingRuleSets___closed__2_value),LEAN_SCALAR_PTR_LITERAL(212, 61, 86, 210, 90, 71, 197, 0)}};
static const lean_ctor_object lp_aesop_Aesop_Frontend_tacticForward_________00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_tacticForward_________00__closed__1_value_aux_1),((lean_object*)&lp_aesop_Aesop_Frontend_tacticForward_________00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(69, 105, 236, 216, 70, 84, 89, 232)}};
static const lean_object* lp_aesop_Aesop_Frontend_tacticForward_________00__closed__1 = (const lean_object*)&lp_aesop_Aesop_Frontend_tacticForward_________00__closed__1_value;
static const lean_string_object lp_aesop_Aesop_Frontend_tacticForward_________00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "forward"};
static const lean_object* lp_aesop_Aesop_Frontend_tacticForward_________00__closed__2 = (const lean_object*)&lp_aesop_Aesop_Frontend_tacticForward_________00__closed__2_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_tacticForward_________00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_tacticForward_________00__closed__2_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_aesop_Aesop_Frontend_tacticForward_________00__closed__3 = (const lean_object*)&lp_aesop_Aesop_Frontend_tacticForward_________00__closed__3_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_tacticForward_________00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_usingRuleSets___closed__6_value),((lean_object*)&lp_aesop_Aesop_Frontend_tacticForward_________00__closed__3_value),((lean_object*)&lp_aesop_Aesop_Frontend_saturate___closed__15_value)}};
static const lean_object* lp_aesop_Aesop_Frontend_tacticForward_________00__closed__4 = (const lean_object*)&lp_aesop_Aesop_Frontend_tacticForward_________00__closed__4_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_tacticForward_________00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_usingRuleSets___closed__6_value),((lean_object*)&lp_aesop_Aesop_Frontend_tacticForward_________00__closed__4_value),((lean_object*)&lp_aesop_Aesop_Frontend_saturate___closed__18_value)}};
static const lean_object* lp_aesop_Aesop_Frontend_tacticForward_________00__closed__5 = (const lean_object*)&lp_aesop_Aesop_Frontend_tacticForward_________00__closed__5_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_tacticForward_________00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_tacticForward_________00__closed__1_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_Frontend_tacticForward_________00__closed__5_value)}};
static const lean_object* lp_aesop_Aesop_Frontend_tacticForward_________00__closed__6 = (const lean_object*)&lp_aesop_Aesop_Frontend_tacticForward_________00__closed__6_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_Frontend_tacticForward________ = (const lean_object*)&lp_aesop_Aesop_Frontend_tacticForward_________00__closed__6_value;
static const lean_array_object lp_aesop_Aesop_Frontend___aux__Aesop__Frontend__Saturate______macroRules__Aesop__Frontend__tacticForward__________1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_aesop_Aesop_Frontend___aux__Aesop__Frontend__Saturate______macroRules__Aesop__Frontend__tacticForward__________1___closed__0 = (const lean_object*)&lp_aesop_Aesop_Frontend___aux__Aesop__Frontend__Saturate______macroRules__Aesop__Frontend__tacticForward__________1___closed__0_value;
static const lean_string_object lp_aesop_Aesop_Frontend___aux__Aesop__Frontend__Saturate______macroRules__Aesop__Frontend__tacticForward__________1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_aesop_Aesop_Frontend___aux__Aesop__Frontend__Saturate______macroRules__Aesop__Frontend__tacticForward__________1___closed__1 = (const lean_object*)&lp_aesop_Aesop_Frontend___aux__Aesop__Frontend__Saturate______macroRules__Aesop__Frontend__tacticForward__________1___closed__1_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend___aux__Aesop__Frontend__Saturate______macroRules__Aesop__Frontend__tacticForward__________1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_Frontend___aux__Aesop__Frontend__Saturate______macroRules__Aesop__Frontend__tacticForward__________1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_aesop_Aesop_Frontend___aux__Aesop__Frontend__Saturate______macroRules__Aesop__Frontend__tacticForward__________1___closed__2 = (const lean_object*)&lp_aesop_Aesop_Frontend___aux__Aesop__Frontend__Saturate______macroRules__Aesop__Frontend__tacticForward__________1___closed__2_value;
static const lean_string_object lp_aesop_Aesop_Frontend___aux__Aesop__Frontend__Saturate______macroRules__Aesop__Frontend__tacticForward__________1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "1"};
static const lean_object* lp_aesop_Aesop_Frontend___aux__Aesop__Frontend__Saturate______macroRules__Aesop__Frontend__tacticForward__________1___closed__3 = (const lean_object*)&lp_aesop_Aesop_Frontend___aux__Aesop__Frontend__Saturate______macroRules__Aesop__Frontend__tacticForward__________1___closed__3_value;
static lean_once_cell_t lp_aesop_Aesop_Frontend___aux__Aesop__Frontend__Saturate______macroRules__Aesop__Frontend__tacticForward__________1___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Frontend___aux__Aesop__Frontend__Saturate______macroRules__Aesop__Frontend__tacticForward__________1___closed__4;
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend___aux__Aesop__Frontend__Saturate______macroRules__Aesop__Frontend__tacticForward__________1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend___aux__Aesop__Frontend__Saturate______macroRules__Aesop__Frontend__tacticForward__________1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop_Aesop_Frontend_tacticForward_x3f_________00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "tacticForward\?____"};
static const lean_object* lp_aesop_Aesop_Frontend_tacticForward_x3f_________00__closed__0 = (const lean_object*)&lp_aesop_Aesop_Frontend_tacticForward_x3f_________00__closed__0_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_tacticForward_x3f_________00__closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_usingRuleSets___closed__1_value),LEAN_SCALAR_PTR_LITERAL(97, 226, 101, 135, 78, 117, 164, 248)}};
static const lean_ctor_object lp_aesop_Aesop_Frontend_tacticForward_x3f_________00__closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_tacticForward_x3f_________00__closed__1_value_aux_0),((lean_object*)&lp_aesop_Aesop_Frontend_Parser_usingRuleSets___closed__2_value),LEAN_SCALAR_PTR_LITERAL(212, 61, 86, 210, 90, 71, 197, 0)}};
static const lean_ctor_object lp_aesop_Aesop_Frontend_tacticForward_x3f_________00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_tacticForward_x3f_________00__closed__1_value_aux_1),((lean_object*)&lp_aesop_Aesop_Frontend_tacticForward_x3f_________00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(14, 253, 145, 1, 108, 22, 225, 237)}};
static const lean_object* lp_aesop_Aesop_Frontend_tacticForward_x3f_________00__closed__1 = (const lean_object*)&lp_aesop_Aesop_Frontend_tacticForward_x3f_________00__closed__1_value;
static const lean_string_object lp_aesop_Aesop_Frontend_tacticForward_x3f_________00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "forward\?"};
static const lean_object* lp_aesop_Aesop_Frontend_tacticForward_x3f_________00__closed__2 = (const lean_object*)&lp_aesop_Aesop_Frontend_tacticForward_x3f_________00__closed__2_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_tacticForward_x3f_________00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_tacticForward_x3f_________00__closed__2_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_aesop_Aesop_Frontend_tacticForward_x3f_________00__closed__3 = (const lean_object*)&lp_aesop_Aesop_Frontend_tacticForward_x3f_________00__closed__3_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_tacticForward_x3f_________00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_usingRuleSets___closed__6_value),((lean_object*)&lp_aesop_Aesop_Frontend_tacticForward_x3f_________00__closed__3_value),((lean_object*)&lp_aesop_Aesop_Frontend_saturate___closed__15_value)}};
static const lean_object* lp_aesop_Aesop_Frontend_tacticForward_x3f_________00__closed__4 = (const lean_object*)&lp_aesop_Aesop_Frontend_tacticForward_x3f_________00__closed__4_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_tacticForward_x3f_________00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_Parser_usingRuleSets___closed__6_value),((lean_object*)&lp_aesop_Aesop_Frontend_tacticForward_x3f_________00__closed__4_value),((lean_object*)&lp_aesop_Aesop_Frontend_saturate___closed__18_value)}};
static const lean_object* lp_aesop_Aesop_Frontend_tacticForward_x3f_________00__closed__5 = (const lean_object*)&lp_aesop_Aesop_Frontend_tacticForward_x3f_________00__closed__5_value;
static const lean_ctor_object lp_aesop_Aesop_Frontend_tacticForward_x3f_________00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Frontend_tacticForward_x3f_________00__closed__1_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_Frontend_tacticForward_x3f_________00__closed__5_value)}};
static const lean_object* lp_aesop_Aesop_Frontend_tacticForward_x3f_________00__closed__6 = (const lean_object*)&lp_aesop_Aesop_Frontend_tacticForward_x3f_________00__closed__6_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_Frontend_tacticForward_x3f________ = (const lean_object*)&lp_aesop_Aesop_Frontend_tacticForward_x3f_________00__closed__6_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend___aux__Aesop__Frontend__Saturate______macroRules__Aesop__Frontend__tacticForward_x3f__________1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend___aux__Aesop__Frontend__Saturate______macroRules__Aesop__Frontend__tacticForward_x3f__________1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ElabM_runForwardElab___redArg(lean_object* v_goal_105_, lean_object* v_x_106_, lean_object* v_a_107_, lean_object* v_a_108_, lean_object* v_a_109_, lean_object* v_a_110_, lean_object* v_a_111_, lean_object* v_a_112_){
_start:
{
uint8_t v___x_114_; lean_object* v___x_115_; lean_object* v___x_116_; 
v___x_114_ = 1;
v___x_115_ = lean_alloc_ctor(0, 1, 1);
lean_ctor_set(v___x_115_, 0, v_goal_105_);
lean_ctor_set_uint8(v___x_115_, sizeof(void*)*1, v___x_114_);
lean_inc(v_a_112_);
lean_inc_ref(v_a_111_);
lean_inc(v_a_110_);
lean_inc_ref(v_a_109_);
lean_inc(v_a_108_);
lean_inc_ref(v_a_107_);
v___x_116_ = lean_apply_8(v_x_106_, v___x_115_, v_a_107_, v_a_108_, v_a_109_, v_a_110_, v_a_111_, v_a_112_, lean_box(0));
return v___x_116_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ElabM_runForwardElab___redArg___boxed(lean_object* v_goal_117_, lean_object* v_x_118_, lean_object* v_a_119_, lean_object* v_a_120_, lean_object* v_a_121_, lean_object* v_a_122_, lean_object* v_a_123_, lean_object* v_a_124_, lean_object* v_a_125_){
_start:
{
lean_object* v_res_126_; 
v_res_126_ = lp_aesop_Aesop_ElabM_runForwardElab___redArg(v_goal_117_, v_x_118_, v_a_119_, v_a_120_, v_a_121_, v_a_122_, v_a_123_, v_a_124_);
lean_dec(v_a_124_);
lean_dec_ref(v_a_123_);
lean_dec(v_a_122_);
lean_dec_ref(v_a_121_);
lean_dec(v_a_120_);
lean_dec_ref(v_a_119_);
return v_res_126_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ElabM_runForwardElab(lean_object* v_00_u03b1_127_, lean_object* v_goal_128_, lean_object* v_x_129_, lean_object* v_a_130_, lean_object* v_a_131_, lean_object* v_a_132_, lean_object* v_a_133_, lean_object* v_a_134_, lean_object* v_a_135_){
_start:
{
lean_object* v___x_137_; 
v___x_137_ = lp_aesop_Aesop_ElabM_runForwardElab___redArg(v_goal_128_, v_x_129_, v_a_130_, v_a_131_, v_a_132_, v_a_133_, v_a_134_, v_a_135_);
return v___x_137_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ElabM_runForwardElab___boxed(lean_object* v_00_u03b1_138_, lean_object* v_goal_139_, lean_object* v_x_140_, lean_object* v_a_141_, lean_object* v_a_142_, lean_object* v_a_143_, lean_object* v_a_144_, lean_object* v_a_145_, lean_object* v_a_146_, lean_object* v_a_147_){
_start:
{
lean_object* v_res_148_; 
v_res_148_ = lp_aesop_Aesop_ElabM_runForwardElab(v_00_u03b1_138_, v_goal_139_, v_x_140_, v_a_141_, v_a_142_, v_a_143_, v_a_144_, v_a_145_, v_a_146_);
lean_dec(v_a_146_);
lean_dec_ref(v_a_145_);
lean_dec(v_a_144_);
lean_dec_ref(v_a_143_);
lean_dec(v_a_142_);
lean_dec_ref(v_a_141_);
return v_res_148_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Check_isEnabled___at___00Aesop_Options_toOptions_x27___at___00Aesop_Frontend_mkForwardOptions_spec__0_spec__1___redArg(lean_object* v_opt_149_, lean_object* v___y_150_){
_start:
{
lean_object* v_options_152_; uint8_t v___x_153_; lean_object* v___x_154_; lean_object* v___x_155_; 
v_options_152_ = lean_ctor_get(v___y_150_, 2);
v___x_153_ = lp_aesop_Aesop_Check_get(v_options_152_, v_opt_149_);
v___x_154_ = lean_box(v___x_153_);
v___x_155_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_155_, 0, v___x_154_);
return v___x_155_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Check_isEnabled___at___00Aesop_Options_toOptions_x27___at___00Aesop_Frontend_mkForwardOptions_spec__0_spec__1___redArg___boxed(lean_object* v_opt_156_, lean_object* v___y_157_, lean_object* v___y_158_){
_start:
{
lean_object* v_res_159_; 
v_res_159_ = lp_aesop_Aesop_Check_isEnabled___at___00Aesop_Options_toOptions_x27___at___00Aesop_Frontend_mkForwardOptions_spec__0_spec__1___redArg(v_opt_156_, v___y_157_);
lean_dec_ref(v___y_157_);
lean_dec_ref(v_opt_156_);
return v_res_159_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Lean_Option_get___at___00Aesop_Options_toOptions_x27___at___00Aesop_Frontend_mkForwardOptions_spec__0_spec__0(lean_object* v_opts_160_, lean_object* v_opt_161_){
_start:
{
lean_object* v_name_162_; lean_object* v_defValue_163_; lean_object* v_map_164_; lean_object* v___x_165_; 
v_name_162_ = lean_ctor_get(v_opt_161_, 0);
v_defValue_163_ = lean_ctor_get(v_opt_161_, 1);
v_map_164_ = lean_ctor_get(v_opts_160_, 0);
v___x_165_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_164_, v_name_162_);
if (lean_obj_tag(v___x_165_) == 0)
{
uint8_t v___x_166_; 
v___x_166_ = lean_unbox(v_defValue_163_);
return v___x_166_;
}
else
{
lean_object* v_val_167_; 
v_val_167_ = lean_ctor_get(v___x_165_, 0);
lean_inc(v_val_167_);
lean_dec_ref_known(v___x_165_, 1);
if (lean_obj_tag(v_val_167_) == 1)
{
uint8_t v_v_168_; 
v_v_168_ = lean_ctor_get_uint8(v_val_167_, 0);
lean_dec_ref_known(v_val_167_, 0);
return v_v_168_;
}
else
{
uint8_t v___x_169_; 
lean_dec(v_val_167_);
v___x_169_ = lean_unbox(v_defValue_163_);
return v___x_169_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_get___at___00Aesop_Options_toOptions_x27___at___00Aesop_Frontend_mkForwardOptions_spec__0_spec__0___boxed(lean_object* v_opts_170_, lean_object* v_opt_171_){
_start:
{
uint8_t v_res_172_; lean_object* v_r_173_; 
v_res_172_ = lp_aesop_Lean_Option_get___at___00Aesop_Options_toOptions_x27___at___00Aesop_Frontend_mkForwardOptions_spec__0_spec__0(v_opts_170_, v_opt_171_);
lean_dec_ref(v_opt_171_);
lean_dec_ref(v_opts_170_);
v_r_173_ = lean_box(v_res_172_);
return v_r_173_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Options_toOptions_x27___at___00Aesop_Frontend_mkForwardOptions_spec__0(lean_object* v_opts_174_, lean_object* v_forwardMaxDepth_x3f_175_, lean_object* v___y_176_, lean_object* v___y_177_){
_start:
{
uint8_t v_a_180_; lean_object* v___y_184_; lean_object* v_options_187_; lean_object* v___x_188_; uint8_t v___x_189_; 
v_options_187_ = lean_ctor_get(v___y_176_, 2);
v___x_188_ = lp_aesop_Aesop_aesop_dev_generateScript;
v___x_189_ = lp_aesop_Lean_Option_get___at___00Aesop_Options_toOptions_x27___at___00Aesop_Frontend_mkForwardOptions_spec__0_spec__0(v_options_187_, v___x_188_);
if (v___x_189_ == 0)
{
uint8_t v_traceScript_190_; 
v_traceScript_190_ = lean_ctor_get_uint8(v_opts_174_, sizeof(void*)*6 + 6);
if (v_traceScript_190_ == 0)
{
lean_object* v___x_191_; lean_object* v___x_192_; lean_object* v_a_193_; uint8_t v___x_194_; 
v___x_191_ = lp_aesop_Aesop_Check_script;
v___x_192_ = lp_aesop_Aesop_Check_isEnabled___at___00Aesop_Options_toOptions_x27___at___00Aesop_Frontend_mkForwardOptions_spec__0_spec__1___redArg(v___x_191_, v___y_176_);
v_a_193_ = lean_ctor_get(v___x_192_, 0);
lean_inc(v_a_193_);
v___x_194_ = lean_unbox(v_a_193_);
lean_dec(v_a_193_);
if (v___x_194_ == 0)
{
lean_object* v___x_195_; lean_object* v___x_196_; 
lean_dec_ref(v___x_192_);
v___x_195_ = lp_aesop_Aesop_Check_script_steps;
v___x_196_ = lp_aesop_Aesop_Check_isEnabled___at___00Aesop_Options_toOptions_x27___at___00Aesop_Frontend_mkForwardOptions_spec__0_spec__1___redArg(v___x_195_, v___y_176_);
v___y_184_ = v___x_196_;
goto v___jp_183_;
}
else
{
v___y_184_ = v___x_192_;
goto v___jp_183_;
}
}
else
{
v_a_180_ = v_traceScript_190_;
goto v___jp_179_;
}
}
else
{
v_a_180_ = v___x_189_;
goto v___jp_179_;
}
v___jp_179_:
{
lean_object* v___x_181_; lean_object* v___x_182_; 
v___x_181_ = lean_alloc_ctor(0, 2, 1);
lean_ctor_set(v___x_181_, 0, v_opts_174_);
lean_ctor_set(v___x_181_, 1, v_forwardMaxDepth_x3f_175_);
lean_ctor_set_uint8(v___x_181_, sizeof(void*)*2, v_a_180_);
v___x_182_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_182_, 0, v___x_181_);
return v___x_182_;
}
v___jp_183_:
{
lean_object* v_a_185_; uint8_t v___x_186_; 
v_a_185_ = lean_ctor_get(v___y_184_, 0);
lean_inc(v_a_185_);
lean_dec_ref(v___y_184_);
v___x_186_ = lean_unbox(v_a_185_);
lean_dec(v_a_185_);
v_a_180_ = v___x_186_;
goto v___jp_179_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Options_toOptions_x27___at___00Aesop_Frontend_mkForwardOptions_spec__0___boxed(lean_object* v_opts_197_, lean_object* v_forwardMaxDepth_x3f_198_, lean_object* v___y_199_, lean_object* v___y_200_, lean_object* v___y_201_){
_start:
{
lean_object* v_res_202_; 
v_res_202_ = lp_aesop_Aesop_Options_toOptions_x27___at___00Aesop_Frontend_mkForwardOptions_spec__0(v_opts_197_, v_forwardMaxDepth_x3f_198_, v___y_199_, v___y_200_);
lean_dec(v___y_200_);
lean_dec_ref(v___y_199_);
return v_res_202_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_mkForwardOptions(lean_object* v_maxDepth_x3f_203_, uint8_t v_traceScript_204_, lean_object* v_a_205_, lean_object* v_a_206_){
_start:
{
uint8_t v___x_208_; lean_object* v___x_209_; lean_object* v___x_210_; lean_object* v___x_211_; lean_object* v___x_212_; lean_object* v___x_213_; uint8_t v___x_214_; uint8_t v___x_215_; lean_object* v___x_216_; uint8_t v___x_217_; uint8_t v___x_218_; lean_object* v___x_219_; lean_object* v___x_220_; 
v___x_208_ = 0;
v___x_209_ = lean_unsigned_to_nat(30u);
v___x_210_ = lean_unsigned_to_nat(200u);
v___x_211_ = lean_unsigned_to_nat(0u);
v___x_212_ = lean_unsigned_to_nat(100u);
v___x_213_ = lean_unsigned_to_nat(50u);
v___x_214_ = 1;
v___x_215_ = 2;
v___x_216_ = lean_box(0);
v___x_217_ = 0;
v___x_218_ = 1;
v___x_219_ = lean_alloc_ctor(0, 6, 11);
lean_ctor_set(v___x_219_, 0, v___x_209_);
lean_ctor_set(v___x_219_, 1, v___x_210_);
lean_ctor_set(v___x_219_, 2, v___x_211_);
lean_ctor_set(v___x_219_, 3, v___x_212_);
lean_ctor_set(v___x_219_, 4, v___x_213_);
lean_ctor_set(v___x_219_, 5, v___x_216_);
lean_ctor_set_uint8(v___x_219_, sizeof(void*)*6, v___x_208_);
lean_ctor_set_uint8(v___x_219_, sizeof(void*)*6 + 1, v___x_214_);
lean_ctor_set_uint8(v___x_219_, sizeof(void*)*6 + 2, v___x_214_);
lean_ctor_set_uint8(v___x_219_, sizeof(void*)*6 + 3, v___x_215_);
lean_ctor_set_uint8(v___x_219_, sizeof(void*)*6 + 4, v___x_217_);
lean_ctor_set_uint8(v___x_219_, sizeof(void*)*6 + 5, v___x_218_);
lean_ctor_set_uint8(v___x_219_, sizeof(void*)*6 + 6, v_traceScript_204_);
lean_ctor_set_uint8(v___x_219_, sizeof(void*)*6 + 7, v___x_218_);
lean_ctor_set_uint8(v___x_219_, sizeof(void*)*6 + 8, v___x_218_);
lean_ctor_set_uint8(v___x_219_, sizeof(void*)*6 + 9, v___x_218_);
lean_ctor_set_uint8(v___x_219_, sizeof(void*)*6 + 10, v___x_218_);
v___x_220_ = lp_aesop_Aesop_Options_toOptions_x27___at___00Aesop_Frontend_mkForwardOptions_spec__0(v___x_219_, v_maxDepth_x3f_203_, v_a_205_, v_a_206_);
return v___x_220_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_mkForwardOptions___boxed(lean_object* v_maxDepth_x3f_221_, lean_object* v_traceScript_222_, lean_object* v_a_223_, lean_object* v_a_224_, lean_object* v_a_225_){
_start:
{
uint8_t v_traceScript_boxed_226_; lean_object* v_res_227_; 
v_traceScript_boxed_226_ = lean_unbox(v_traceScript_222_);
v_res_227_ = lp_aesop_Aesop_Frontend_mkForwardOptions(v_maxDepth_x3f_221_, v_traceScript_boxed_226_, v_a_223_, v_a_224_);
lean_dec(v_a_224_);
lean_dec_ref(v_a_223_);
return v_res_227_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Check_isEnabled___at___00Aesop_Options_toOptions_x27___at___00Aesop_Frontend_mkForwardOptions_spec__0_spec__1(lean_object* v_opt_228_, lean_object* v___y_229_, lean_object* v___y_230_){
_start:
{
lean_object* v___x_232_; 
v___x_232_ = lp_aesop_Aesop_Check_isEnabled___at___00Aesop_Options_toOptions_x27___at___00Aesop_Frontend_mkForwardOptions_spec__0_spec__1___redArg(v_opt_228_, v___y_229_);
return v___x_232_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Check_isEnabled___at___00Aesop_Options_toOptions_x27___at___00Aesop_Frontend_mkForwardOptions_spec__0_spec__1___boxed(lean_object* v_opt_233_, lean_object* v___y_234_, lean_object* v___y_235_, lean_object* v___y_236_){
_start:
{
lean_object* v_res_237_; 
v_res_237_ = lp_aesop_Aesop_Check_isEnabled___at___00Aesop_Options_toOptions_x27___at___00Aesop_Frontend_mkForwardOptions_spec__0_spec__1(v_opt_233_, v___y_234_, v___y_235_);
lean_dec(v___y_235_);
lean_dec_ref(v___y_234_);
lean_dec_ref(v_opt_233_);
return v_res_237_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00Aesop_Frontend_elabGlobalRuleSets_spec__1(lean_object* v_x_238_, lean_object* v_x_239_){
_start:
{
if (lean_obj_tag(v_x_239_) == 0)
{
return v_x_238_;
}
else
{
lean_object* v_key_240_; lean_object* v_tail_241_; lean_object* v___x_242_; 
v_key_240_ = lean_ctor_get(v_x_239_, 0);
lean_inc(v_key_240_);
v_tail_241_ = lean_ctor_get(v_x_239_, 2);
lean_inc(v_tail_241_);
lean_dec_ref_known(v_x_239_, 3);
v___x_242_ = lean_array_push(v_x_238_, v_key_240_);
v_x_238_ = v___x_242_;
v_x_239_ = v_tail_241_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Frontend_elabGlobalRuleSets_spec__2(lean_object* v_as_244_, size_t v_i_245_, size_t v_stop_246_, lean_object* v_b_247_){
_start:
{
uint8_t v___x_248_; 
v___x_248_ = lean_usize_dec_eq(v_i_245_, v_stop_246_);
if (v___x_248_ == 0)
{
lean_object* v___x_249_; lean_object* v___x_250_; size_t v___x_251_; size_t v___x_252_; 
v___x_249_ = lean_array_uget_borrowed(v_as_244_, v_i_245_);
lean_inc(v___x_249_);
v___x_250_ = lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00Aesop_Frontend_elabGlobalRuleSets_spec__1(v_b_247_, v___x_249_);
v___x_251_ = ((size_t)1ULL);
v___x_252_ = lean_usize_add(v_i_245_, v___x_251_);
v_i_245_ = v___x_252_;
v_b_247_ = v___x_250_;
goto _start;
}
else
{
return v_b_247_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Frontend_elabGlobalRuleSets_spec__2___boxed(lean_object* v_as_254_, lean_object* v_i_255_, lean_object* v_stop_256_, lean_object* v_b_257_){
_start:
{
size_t v_i_boxed_258_; size_t v_stop_boxed_259_; lean_object* v_res_260_; 
v_i_boxed_258_ = lean_unbox_usize(v_i_255_);
lean_dec(v_i_255_);
v_stop_boxed_259_ = lean_unbox_usize(v_stop_256_);
lean_dec(v_stop_256_);
v_res_260_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Frontend_elabGlobalRuleSets_spec__2(v_as_254_, v_i_boxed_258_, v_stop_boxed_259_, v_b_257_);
lean_dec_ref(v_as_254_);
return v_res_260_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Frontend_elabGlobalRuleSets_spec__0(size_t v_sz_261_, size_t v_i_262_, lean_object* v_bs_263_){
_start:
{
uint8_t v___x_264_; 
v___x_264_ = lean_usize_dec_lt(v_i_262_, v_sz_261_);
if (v___x_264_ == 0)
{
return v_bs_263_;
}
else
{
lean_object* v_v_265_; lean_object* v___x_266_; lean_object* v_bs_x27_267_; lean_object* v___x_268_; size_t v___x_269_; size_t v___x_270_; lean_object* v___x_271_; 
v_v_265_ = lean_array_uget(v_bs_263_, v_i_262_);
v___x_266_ = lean_unsigned_to_nat(0u);
v_bs_x27_267_ = lean_array_uset(v_bs_263_, v_i_262_, v___x_266_);
v___x_268_ = l_Lean_TSyntax_getId(v_v_265_);
lean_dec(v_v_265_);
v___x_269_ = ((size_t)1ULL);
v___x_270_ = lean_usize_add(v_i_262_, v___x_269_);
v___x_271_ = lean_array_uset(v_bs_x27_267_, v_i_262_, v___x_268_);
v_i_262_ = v___x_270_;
v_bs_263_ = v___x_271_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Frontend_elabGlobalRuleSets_spec__0___boxed(lean_object* v_sz_273_, lean_object* v_i_274_, lean_object* v_bs_275_){
_start:
{
size_t v_sz_boxed_276_; size_t v_i_boxed_277_; lean_object* v_res_278_; 
v_sz_boxed_276_ = lean_unbox_usize(v_sz_273_);
lean_dec(v_sz_273_);
v_i_boxed_277_ = lean_unbox_usize(v_i_274_);
lean_dec(v_i_274_);
v_res_278_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Frontend_elabGlobalRuleSets_spec__0(v_sz_boxed_276_, v_i_boxed_277_, v_bs_275_);
return v_res_278_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_elabGlobalRuleSets(lean_object* v_rsNames_279_, lean_object* v_a_280_, lean_object* v_a_281_){
_start:
{
lean_object* v___x_283_; 
v___x_283_ = lp_aesop_Aesop_getDefaultRuleSetNames();
if (lean_obj_tag(v___x_283_) == 0)
{
lean_object* v_a_284_; lean_object* v___y_286_; lean_object* v_size_292_; lean_object* v_buckets_293_; lean_object* v___x_294_; lean_object* v___x_295_; lean_object* v___x_296_; uint8_t v___x_297_; 
v_a_284_ = lean_ctor_get(v___x_283_, 0);
lean_inc(v_a_284_);
lean_dec_ref_known(v___x_283_, 1);
v_size_292_ = lean_ctor_get(v_a_284_, 0);
lean_inc(v_size_292_);
v_buckets_293_ = lean_ctor_get(v_a_284_, 1);
lean_inc_ref(v_buckets_293_);
lean_dec(v_a_284_);
v___x_294_ = lean_mk_empty_array_with_capacity(v_size_292_);
lean_dec(v_size_292_);
v___x_295_ = lean_unsigned_to_nat(0u);
v___x_296_ = lean_array_get_size(v_buckets_293_);
v___x_297_ = lean_nat_dec_lt(v___x_295_, v___x_296_);
if (v___x_297_ == 0)
{
lean_dec_ref(v_buckets_293_);
v___y_286_ = v___x_294_;
goto v___jp_285_;
}
else
{
uint8_t v___x_298_; 
v___x_298_ = lean_nat_dec_le(v___x_296_, v___x_296_);
if (v___x_298_ == 0)
{
if (v___x_297_ == 0)
{
lean_dec_ref(v_buckets_293_);
v___y_286_ = v___x_294_;
goto v___jp_285_;
}
else
{
size_t v___x_299_; size_t v___x_300_; lean_object* v___x_301_; 
v___x_299_ = ((size_t)0ULL);
v___x_300_ = lean_usize_of_nat(v___x_296_);
v___x_301_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Frontend_elabGlobalRuleSets_spec__2(v_buckets_293_, v___x_299_, v___x_300_, v___x_294_);
lean_dec_ref(v_buckets_293_);
v___y_286_ = v___x_301_;
goto v___jp_285_;
}
}
else
{
size_t v___x_302_; size_t v___x_303_; lean_object* v___x_304_; 
v___x_302_ = ((size_t)0ULL);
v___x_303_ = lean_usize_of_nat(v___x_296_);
v___x_304_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Frontend_elabGlobalRuleSets_spec__2(v_buckets_293_, v___x_302_, v___x_303_, v___x_294_);
lean_dec_ref(v_buckets_293_);
v___y_286_ = v___x_304_;
goto v___jp_285_;
}
}
v___jp_285_:
{
size_t v_sz_287_; size_t v___x_288_; lean_object* v___x_289_; lean_object* v___x_290_; lean_object* v___x_291_; 
v_sz_287_ = lean_array_size(v_rsNames_279_);
v___x_288_ = ((size_t)0ULL);
v___x_289_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Frontend_elabGlobalRuleSets_spec__0(v_sz_287_, v___x_288_, v_rsNames_279_);
v___x_290_ = l_Array_append___redArg(v___y_286_, v___x_289_);
lean_dec_ref(v___x_289_);
v___x_291_ = lp_aesop_Aesop_Frontend_getGlobalRuleSets(v___x_290_, v_a_280_, v_a_281_);
return v___x_291_;
}
}
else
{
lean_object* v_a_305_; lean_object* v___x_307_; uint8_t v_isShared_308_; uint8_t v_isSharedCheck_317_; 
lean_dec_ref(v_rsNames_279_);
v_a_305_ = lean_ctor_get(v___x_283_, 0);
v_isSharedCheck_317_ = !lean_is_exclusive(v___x_283_);
if (v_isSharedCheck_317_ == 0)
{
v___x_307_ = v___x_283_;
v_isShared_308_ = v_isSharedCheck_317_;
goto v_resetjp_306_;
}
else
{
lean_inc(v_a_305_);
lean_dec(v___x_283_);
v___x_307_ = lean_box(0);
v_isShared_308_ = v_isSharedCheck_317_;
goto v_resetjp_306_;
}
v_resetjp_306_:
{
lean_object* v_ref_309_; lean_object* v___x_310_; lean_object* v___x_311_; lean_object* v___x_312_; lean_object* v___x_313_; lean_object* v___x_315_; 
v_ref_309_ = lean_ctor_get(v_a_280_, 5);
v___x_310_ = lean_io_error_to_string(v_a_305_);
v___x_311_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_311_, 0, v___x_310_);
v___x_312_ = l_Lean_MessageData_ofFormat(v___x_311_);
lean_inc(v_ref_309_);
v___x_313_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_313_, 0, v_ref_309_);
lean_ctor_set(v___x_313_, 1, v___x_312_);
if (v_isShared_308_ == 0)
{
lean_ctor_set(v___x_307_, 0, v___x_313_);
v___x_315_ = v___x_307_;
goto v_reusejp_314_;
}
else
{
lean_object* v_reuseFailAlloc_316_; 
v_reuseFailAlloc_316_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_316_, 0, v___x_313_);
v___x_315_ = v_reuseFailAlloc_316_;
goto v_reusejp_314_;
}
v_reusejp_314_:
{
return v___x_315_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_elabGlobalRuleSets___boxed(lean_object* v_rsNames_318_, lean_object* v_a_319_, lean_object* v_a_320_, lean_object* v_a_321_){
_start:
{
lean_object* v_res_322_; 
v_res_322_ = lp_aesop_Aesop_Frontend_elabGlobalRuleSets(v_rsNames_318_, v_a_319_, v_a_320_);
lean_dec(v_a_320_);
lean_dec_ref(v_a_319_);
return v_res_322_;
}
}
static lean_object* _init_lp_aesop_Aesop_Frontend_elabForwardRule___closed__0(void){
_start:
{
lean_object* v___x_323_; lean_object* v___x_324_; 
v___x_323_ = lean_unsigned_to_nat(1u);
v___x_324_ = lean_nat_to_int(v___x_323_);
return v___x_324_;
}
}
static lean_object* _init_lp_aesop_Aesop_Frontend_elabForwardRule___closed__1(void){
_start:
{
uint8_t v___x_325_; lean_object* v___x_326_; lean_object* v___x_327_; 
v___x_325_ = 0;
v___x_326_ = lean_obj_once(&lp_aesop_Aesop_Frontend_elabForwardRule___closed__0, &lp_aesop_Aesop_Frontend_elabForwardRule___closed__0_once, _init_lp_aesop_Aesop_Frontend_elabForwardRule___closed__0);
v___x_327_ = lean_alloc_ctor(0, 1, 1);
lean_ctor_set(v___x_327_, 0, v___x_326_);
lean_ctor_set_uint8(v___x_327_, sizeof(void*)*1, v___x_325_);
return v___x_327_;
}
}
static lean_object* _init_lp_aesop_Aesop_Frontend_elabForwardRule___closed__2(void){
_start:
{
lean_object* v___x_328_; lean_object* v___x_329_; 
v___x_328_ = lean_obj_once(&lp_aesop_Aesop_Frontend_elabForwardRule___closed__1, &lp_aesop_Aesop_Frontend_elabForwardRule___closed__1_once, _init_lp_aesop_Aesop_Frontend_elabForwardRule___closed__1);
v___x_329_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_329_, 0, v___x_328_);
return v___x_329_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_elabForwardRule(lean_object* v_term_330_, lean_object* v_a_331_, lean_object* v_a_332_, lean_object* v_a_333_, lean_object* v_a_334_, lean_object* v_a_335_, lean_object* v_a_336_, lean_object* v_a_337_){
_start:
{
lean_object* v_goal_339_; lean_object* v___x_340_; lean_object* v___x_341_; lean_object* v_builderInput_342_; uint8_t v___x_343_; lean_object* v___x_344_; uint8_t v___x_345_; lean_object* v___x_346_; 
v_goal_339_ = lean_ctor_get(v_a_331_, 0);
v___x_340_ = lp_aesop_Aesop_RuleBuilderOptions_default;
v___x_341_ = lean_obj_once(&lp_aesop_Aesop_Frontend_elabForwardRule___closed__2, &lp_aesop_Aesop_Frontend_elabForwardRule___closed__2_once, _init_lp_aesop_Aesop_Frontend_elabForwardRule___closed__2);
v_builderInput_342_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_builderInput_342_, 0, v_term_330_);
lean_ctor_set(v_builderInput_342_, 1, v___x_340_);
lean_ctor_set(v_builderInput_342_, 2, v___x_341_);
v___x_343_ = 1;
lean_inc(v_goal_339_);
v___x_344_ = lean_alloc_ctor(0, 1, 1);
lean_ctor_set(v___x_344_, 0, v_goal_339_);
lean_ctor_set_uint8(v___x_344_, sizeof(void*)*1, v___x_343_);
v___x_345_ = 0;
v___x_346_ = lp_aesop_Aesop_RuleBuilder_forward(v___x_345_, v_builderInput_342_, v___x_344_, v_a_332_, v_a_333_, v_a_334_, v_a_335_, v_a_336_, v_a_337_);
lean_dec_ref_known(v___x_344_, 1);
return v___x_346_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_elabForwardRule___boxed(lean_object* v_term_347_, lean_object* v_a_348_, lean_object* v_a_349_, lean_object* v_a_350_, lean_object* v_a_351_, lean_object* v_a_352_, lean_object* v_a_353_, lean_object* v_a_354_, lean_object* v_a_355_){
_start:
{
lean_object* v_res_356_; 
v_res_356_ = lp_aesop_Aesop_Frontend_elabForwardRule(v_term_347_, v_a_348_, v_a_349_, v_a_350_, v_a_351_, v_a_352_, v_a_353_, v_a_354_);
lean_dec(v_a_354_);
lean_dec_ref(v_a_353_);
lean_dec(v_a_352_);
lean_dec_ref(v_a_351_);
lean_dec(v_a_350_);
lean_dec_ref(v_a_349_);
lean_dec_ref(v_a_348_);
return v_res_356_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_mkHypForwardRule(lean_object* v_fvarId_357_, lean_object* v_a_358_, lean_object* v_a_359_, lean_object* v_a_360_, lean_object* v_a_361_, lean_object* v_a_362_, lean_object* v_a_363_, lean_object* v_a_364_){
_start:
{
lean_object* v___x_366_; lean_object* v___x_367_; lean_object* v___x_368_; 
v___x_366_ = l_Lean_Expr_fvar___override(v_fvarId_357_);
v___x_367_ = lean_box(1);
v___x_368_ = l_Lean_PrettyPrinter_delab(v___x_366_, v___x_367_, v_a_361_, v_a_362_, v_a_363_, v_a_364_);
if (lean_obj_tag(v___x_368_) == 0)
{
lean_object* v_a_369_; lean_object* v___x_370_; 
v_a_369_ = lean_ctor_get(v___x_368_, 0);
lean_inc(v_a_369_);
lean_dec_ref_known(v___x_368_, 1);
v___x_370_ = lp_aesop_Aesop_Frontend_elabForwardRule(v_a_369_, v_a_358_, v_a_359_, v_a_360_, v_a_361_, v_a_362_, v_a_363_, v_a_364_);
return v___x_370_;
}
else
{
lean_object* v_a_371_; lean_object* v___x_373_; uint8_t v_isShared_374_; uint8_t v_isSharedCheck_378_; 
v_a_371_ = lean_ctor_get(v___x_368_, 0);
v_isSharedCheck_378_ = !lean_is_exclusive(v___x_368_);
if (v_isSharedCheck_378_ == 0)
{
v___x_373_ = v___x_368_;
v_isShared_374_ = v_isSharedCheck_378_;
goto v_resetjp_372_;
}
else
{
lean_inc(v_a_371_);
lean_dec(v___x_368_);
v___x_373_ = lean_box(0);
v_isShared_374_ = v_isSharedCheck_378_;
goto v_resetjp_372_;
}
v_resetjp_372_:
{
lean_object* v___x_376_; 
if (v_isShared_374_ == 0)
{
v___x_376_ = v___x_373_;
goto v_reusejp_375_;
}
else
{
lean_object* v_reuseFailAlloc_377_; 
v_reuseFailAlloc_377_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_377_, 0, v_a_371_);
v___x_376_ = v_reuseFailAlloc_377_;
goto v_reusejp_375_;
}
v_reusejp_375_:
{
return v___x_376_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_mkHypForwardRule___boxed(lean_object* v_fvarId_379_, lean_object* v_a_380_, lean_object* v_a_381_, lean_object* v_a_382_, lean_object* v_a_383_, lean_object* v_a_384_, lean_object* v_a_385_, lean_object* v_a_386_, lean_object* v_a_387_){
_start:
{
lean_object* v_res_388_; 
v_res_388_ = lp_aesop_Aesop_Frontend_mkHypForwardRule(v_fvarId_379_, v_a_380_, v_a_381_, v_a_382_, v_a_383_, v_a_384_, v_a_385_, v_a_386_);
lean_dec(v_a_386_);
lean_dec_ref(v_a_385_);
lean_dec(v_a_384_);
lean_dec_ref(v_a_383_);
lean_dec(v_a_382_);
lean_dec_ref(v_a_381_);
lean_dec_ref(v_a_380_);
return v_res_388_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_forallBoundedTelescope___at___00Aesop_Frontend_isImplication_spec__0___redArg___lam__0(lean_object* v_k_389_, lean_object* v_b_390_, lean_object* v_c_391_, lean_object* v___y_392_, lean_object* v___y_393_, lean_object* v___y_394_, lean_object* v___y_395_){
_start:
{
lean_object* v___x_397_; 
lean_inc(v___y_395_);
lean_inc_ref(v___y_394_);
lean_inc(v___y_393_);
lean_inc_ref(v___y_392_);
v___x_397_ = lean_apply_7(v_k_389_, v_b_390_, v_c_391_, v___y_392_, v___y_393_, v___y_394_, v___y_395_, lean_box(0));
return v___x_397_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_forallBoundedTelescope___at___00Aesop_Frontend_isImplication_spec__0___redArg___lam__0___boxed(lean_object* v_k_398_, lean_object* v_b_399_, lean_object* v_c_400_, lean_object* v___y_401_, lean_object* v___y_402_, lean_object* v___y_403_, lean_object* v___y_404_, lean_object* v___y_405_){
_start:
{
lean_object* v_res_406_; 
v_res_406_ = lp_aesop_Lean_Meta_forallBoundedTelescope___at___00Aesop_Frontend_isImplication_spec__0___redArg___lam__0(v_k_398_, v_b_399_, v_c_400_, v___y_401_, v___y_402_, v___y_403_, v___y_404_);
lean_dec(v___y_404_);
lean_dec_ref(v___y_403_);
lean_dec(v___y_402_);
lean_dec_ref(v___y_401_);
return v_res_406_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_forallBoundedTelescope___at___00Aesop_Frontend_isImplication_spec__0___redArg(lean_object* v_type_407_, lean_object* v_maxFVars_x3f_408_, lean_object* v_k_409_, uint8_t v_cleanupAnnotations_410_, uint8_t v_whnfType_411_, lean_object* v___y_412_, lean_object* v___y_413_, lean_object* v___y_414_, lean_object* v___y_415_){
_start:
{
lean_object* v___f_417_; lean_object* v___x_418_; 
v___f_417_ = lean_alloc_closure((void*)(lp_aesop_Lean_Meta_forallBoundedTelescope___at___00Aesop_Frontend_isImplication_spec__0___redArg___lam__0___boxed), 8, 1);
lean_closure_set(v___f_417_, 0, v_k_409_);
v___x_418_ = l___private_Lean_Meta_Basic_0__Lean_Meta_forallTelescopeReducingAux(lean_box(0), v_type_407_, v_maxFVars_x3f_408_, v___f_417_, v_cleanupAnnotations_410_, v_whnfType_411_, v___y_412_, v___y_413_, v___y_414_, v___y_415_);
if (lean_obj_tag(v___x_418_) == 0)
{
lean_object* v_a_419_; lean_object* v___x_421_; uint8_t v_isShared_422_; uint8_t v_isSharedCheck_426_; 
v_a_419_ = lean_ctor_get(v___x_418_, 0);
v_isSharedCheck_426_ = !lean_is_exclusive(v___x_418_);
if (v_isSharedCheck_426_ == 0)
{
v___x_421_ = v___x_418_;
v_isShared_422_ = v_isSharedCheck_426_;
goto v_resetjp_420_;
}
else
{
lean_inc(v_a_419_);
lean_dec(v___x_418_);
v___x_421_ = lean_box(0);
v_isShared_422_ = v_isSharedCheck_426_;
goto v_resetjp_420_;
}
v_resetjp_420_:
{
lean_object* v___x_424_; 
if (v_isShared_422_ == 0)
{
v___x_424_ = v___x_421_;
goto v_reusejp_423_;
}
else
{
lean_object* v_reuseFailAlloc_425_; 
v_reuseFailAlloc_425_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_425_, 0, v_a_419_);
v___x_424_ = v_reuseFailAlloc_425_;
goto v_reusejp_423_;
}
v_reusejp_423_:
{
return v___x_424_;
}
}
}
else
{
lean_object* v_a_427_; lean_object* v___x_429_; uint8_t v_isShared_430_; uint8_t v_isSharedCheck_434_; 
v_a_427_ = lean_ctor_get(v___x_418_, 0);
v_isSharedCheck_434_ = !lean_is_exclusive(v___x_418_);
if (v_isSharedCheck_434_ == 0)
{
v___x_429_ = v___x_418_;
v_isShared_430_ = v_isSharedCheck_434_;
goto v_resetjp_428_;
}
else
{
lean_inc(v_a_427_);
lean_dec(v___x_418_);
v___x_429_ = lean_box(0);
v_isShared_430_ = v_isSharedCheck_434_;
goto v_resetjp_428_;
}
v_resetjp_428_:
{
lean_object* v___x_432_; 
if (v_isShared_430_ == 0)
{
v___x_432_ = v___x_429_;
goto v_reusejp_431_;
}
else
{
lean_object* v_reuseFailAlloc_433_; 
v_reuseFailAlloc_433_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_433_, 0, v_a_427_);
v___x_432_ = v_reuseFailAlloc_433_;
goto v_reusejp_431_;
}
v_reusejp_431_:
{
return v___x_432_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_forallBoundedTelescope___at___00Aesop_Frontend_isImplication_spec__0___redArg___boxed(lean_object* v_type_435_, lean_object* v_maxFVars_x3f_436_, lean_object* v_k_437_, lean_object* v_cleanupAnnotations_438_, lean_object* v_whnfType_439_, lean_object* v___y_440_, lean_object* v___y_441_, lean_object* v___y_442_, lean_object* v___y_443_, lean_object* v___y_444_){
_start:
{
uint8_t v_cleanupAnnotations_boxed_445_; uint8_t v_whnfType_boxed_446_; lean_object* v_res_447_; 
v_cleanupAnnotations_boxed_445_ = lean_unbox(v_cleanupAnnotations_438_);
v_whnfType_boxed_446_ = lean_unbox(v_whnfType_439_);
v_res_447_ = lp_aesop_Lean_Meta_forallBoundedTelescope___at___00Aesop_Frontend_isImplication_spec__0___redArg(v_type_435_, v_maxFVars_x3f_436_, v_k_437_, v_cleanupAnnotations_boxed_445_, v_whnfType_boxed_446_, v___y_440_, v___y_441_, v___y_442_, v___y_443_);
lean_dec(v___y_443_);
lean_dec_ref(v___y_442_);
lean_dec(v___y_441_);
lean_dec_ref(v___y_440_);
return v_res_447_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_forallBoundedTelescope___at___00Aesop_Frontend_isImplication_spec__0(lean_object* v_00_u03b1_448_, lean_object* v_type_449_, lean_object* v_maxFVars_x3f_450_, lean_object* v_k_451_, uint8_t v_cleanupAnnotations_452_, uint8_t v_whnfType_453_, lean_object* v___y_454_, lean_object* v___y_455_, lean_object* v___y_456_, lean_object* v___y_457_){
_start:
{
lean_object* v___x_459_; 
v___x_459_ = lp_aesop_Lean_Meta_forallBoundedTelescope___at___00Aesop_Frontend_isImplication_spec__0___redArg(v_type_449_, v_maxFVars_x3f_450_, v_k_451_, v_cleanupAnnotations_452_, v_whnfType_453_, v___y_454_, v___y_455_, v___y_456_, v___y_457_);
return v___x_459_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_forallBoundedTelescope___at___00Aesop_Frontend_isImplication_spec__0___boxed(lean_object* v_00_u03b1_460_, lean_object* v_type_461_, lean_object* v_maxFVars_x3f_462_, lean_object* v_k_463_, lean_object* v_cleanupAnnotations_464_, lean_object* v_whnfType_465_, lean_object* v___y_466_, lean_object* v___y_467_, lean_object* v___y_468_, lean_object* v___y_469_, lean_object* v___y_470_){
_start:
{
uint8_t v_cleanupAnnotations_boxed_471_; uint8_t v_whnfType_boxed_472_; lean_object* v_res_473_; 
v_cleanupAnnotations_boxed_471_ = lean_unbox(v_cleanupAnnotations_464_);
v_whnfType_boxed_472_ = lean_unbox(v_whnfType_465_);
v_res_473_ = lp_aesop_Lean_Meta_forallBoundedTelescope___at___00Aesop_Frontend_isImplication_spec__0(v_00_u03b1_460_, v_type_461_, v_maxFVars_x3f_462_, v_k_463_, v_cleanupAnnotations_boxed_471_, v_whnfType_boxed_472_, v___y_466_, v___y_467_, v___y_468_, v___y_469_);
lean_dec(v___y_469_);
lean_dec_ref(v___y_468_);
lean_dec(v___y_467_);
lean_dec_ref(v___y_466_);
return v_res_473_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_isImplication___lam__0(lean_object* v_args_474_, lean_object* v_body_475_, lean_object* v___y_476_, lean_object* v___y_477_, lean_object* v___y_478_, lean_object* v___y_479_){
_start:
{
lean_object* v___x_481_; lean_object* v___x_482_; uint8_t v___x_483_; 
v___x_481_ = lean_array_get_size(v_args_474_);
v___x_482_ = lean_unsigned_to_nat(0u);
v___x_483_ = lean_nat_dec_eq(v___x_481_, v___x_482_);
if (v___x_483_ == 0)
{
lean_object* v___x_484_; 
v___x_484_ = l_Lean_Meta_isProp(v_body_475_, v___y_476_, v___y_477_, v___y_478_, v___y_479_);
return v___x_484_;
}
else
{
uint8_t v___x_485_; lean_object* v___x_486_; lean_object* v___x_487_; 
lean_dec_ref(v_body_475_);
v___x_485_ = 0;
v___x_486_ = lean_box(v___x_485_);
v___x_487_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_487_, 0, v___x_486_);
return v___x_487_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_isImplication___lam__0___boxed(lean_object* v_args_488_, lean_object* v_body_489_, lean_object* v___y_490_, lean_object* v___y_491_, lean_object* v___y_492_, lean_object* v___y_493_, lean_object* v___y_494_){
_start:
{
lean_object* v_res_495_; 
v_res_495_ = lp_aesop_Aesop_Frontend_isImplication___lam__0(v_args_488_, v_body_489_, v___y_490_, v___y_491_, v___y_492_, v___y_493_);
lean_dec(v___y_493_);
lean_dec_ref(v___y_492_);
lean_dec(v___y_491_);
lean_dec_ref(v___y_490_);
lean_dec_ref(v_args_488_);
return v_res_495_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_isImplication(lean_object* v_e_499_, lean_object* v_a_500_, lean_object* v_a_501_, lean_object* v_a_502_, lean_object* v_a_503_){
_start:
{
lean_object* v___f_505_; lean_object* v___x_506_; uint8_t v___x_507_; lean_object* v___x_508_; 
v___f_505_ = ((lean_object*)(lp_aesop_Aesop_Frontend_isImplication___closed__0));
v___x_506_ = ((lean_object*)(lp_aesop_Aesop_Frontend_isImplication___closed__1));
v___x_507_ = 0;
v___x_508_ = lp_aesop_Lean_Meta_forallBoundedTelescope___at___00Aesop_Frontend_isImplication_spec__0___redArg(v_e_499_, v___x_506_, v___f_505_, v___x_507_, v___x_507_, v_a_500_, v_a_501_, v_a_502_, v_a_503_);
return v___x_508_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_isImplication___boxed(lean_object* v_e_509_, lean_object* v_a_510_, lean_object* v_a_511_, lean_object* v_a_512_, lean_object* v_a_513_, lean_object* v_a_514_){
_start:
{
lean_object* v_res_515_; 
v_res_515_ = lp_aesop_Aesop_Frontend_isImplication(v_e_509_, v_a_510_, v_a_511_, v_a_512_, v_a_513_);
lean_dec(v_a_513_);
lean_dec_ref(v_a_512_);
lean_dec(v_a_511_);
lean_dec_ref(v_a_510_);
return v_res_515_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_Frontend_mkHypImplicationRule_x3f_spec__0___redArg___lam__0(lean_object* v_x_516_, lean_object* v___y_517_, lean_object* v___y_518_, lean_object* v___y_519_, lean_object* v___y_520_, lean_object* v___y_521_, lean_object* v___y_522_, lean_object* v___y_523_){
_start:
{
lean_object* v___x_525_; 
lean_inc(v___y_519_);
lean_inc_ref(v___y_518_);
lean_inc_ref(v___y_517_);
v___x_525_ = lean_apply_8(v_x_516_, v___y_517_, v___y_518_, v___y_519_, v___y_520_, v___y_521_, v___y_522_, v___y_523_, lean_box(0));
return v___x_525_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_Frontend_mkHypImplicationRule_x3f_spec__0___redArg___lam__0___boxed(lean_object* v_x_526_, lean_object* v___y_527_, lean_object* v___y_528_, lean_object* v___y_529_, lean_object* v___y_530_, lean_object* v___y_531_, lean_object* v___y_532_, lean_object* v___y_533_, lean_object* v___y_534_){
_start:
{
lean_object* v_res_535_; 
v_res_535_ = lp_aesop_Lean_MVarId_withContext___at___00Aesop_Frontend_mkHypImplicationRule_x3f_spec__0___redArg___lam__0(v_x_526_, v___y_527_, v___y_528_, v___y_529_, v___y_530_, v___y_531_, v___y_532_, v___y_533_);
lean_dec(v___y_529_);
lean_dec_ref(v___y_528_);
lean_dec_ref(v___y_527_);
return v_res_535_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_Frontend_mkHypImplicationRule_x3f_spec__0___redArg(lean_object* v_mvarId_536_, lean_object* v_x_537_, lean_object* v___y_538_, lean_object* v___y_539_, lean_object* v___y_540_, lean_object* v___y_541_, lean_object* v___y_542_, lean_object* v___y_543_, lean_object* v___y_544_){
_start:
{
lean_object* v___f_546_; lean_object* v___x_547_; 
lean_inc(v___y_540_);
lean_inc_ref(v___y_539_);
lean_inc_ref(v___y_538_);
v___f_546_ = lean_alloc_closure((void*)(lp_aesop_Lean_MVarId_withContext___at___00Aesop_Frontend_mkHypImplicationRule_x3f_spec__0___redArg___lam__0___boxed), 9, 4);
lean_closure_set(v___f_546_, 0, v_x_537_);
lean_closure_set(v___f_546_, 1, v___y_538_);
lean_closure_set(v___f_546_, 2, v___y_539_);
lean_closure_set(v___f_546_, 3, v___y_540_);
v___x_547_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withMVarContextImp(lean_box(0), v_mvarId_536_, v___f_546_, v___y_541_, v___y_542_, v___y_543_, v___y_544_);
if (lean_obj_tag(v___x_547_) == 0)
{
return v___x_547_;
}
else
{
lean_object* v_a_548_; lean_object* v___x_550_; uint8_t v_isShared_551_; uint8_t v_isSharedCheck_555_; 
v_a_548_ = lean_ctor_get(v___x_547_, 0);
v_isSharedCheck_555_ = !lean_is_exclusive(v___x_547_);
if (v_isSharedCheck_555_ == 0)
{
v___x_550_ = v___x_547_;
v_isShared_551_ = v_isSharedCheck_555_;
goto v_resetjp_549_;
}
else
{
lean_inc(v_a_548_);
lean_dec(v___x_547_);
v___x_550_ = lean_box(0);
v_isShared_551_ = v_isSharedCheck_555_;
goto v_resetjp_549_;
}
v_resetjp_549_:
{
lean_object* v___x_553_; 
if (v_isShared_551_ == 0)
{
v___x_553_ = v___x_550_;
goto v_reusejp_552_;
}
else
{
lean_object* v_reuseFailAlloc_554_; 
v_reuseFailAlloc_554_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_554_, 0, v_a_548_);
v___x_553_ = v_reuseFailAlloc_554_;
goto v_reusejp_552_;
}
v_reusejp_552_:
{
return v___x_553_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_Frontend_mkHypImplicationRule_x3f_spec__0___redArg___boxed(lean_object* v_mvarId_556_, lean_object* v_x_557_, lean_object* v___y_558_, lean_object* v___y_559_, lean_object* v___y_560_, lean_object* v___y_561_, lean_object* v___y_562_, lean_object* v___y_563_, lean_object* v___y_564_, lean_object* v___y_565_){
_start:
{
lean_object* v_res_566_; 
v_res_566_ = lp_aesop_Lean_MVarId_withContext___at___00Aesop_Frontend_mkHypImplicationRule_x3f_spec__0___redArg(v_mvarId_556_, v_x_557_, v___y_558_, v___y_559_, v___y_560_, v___y_561_, v___y_562_, v___y_563_, v___y_564_);
lean_dec(v___y_564_);
lean_dec_ref(v___y_563_);
lean_dec(v___y_562_);
lean_dec_ref(v___y_561_);
lean_dec(v___y_560_);
lean_dec_ref(v___y_559_);
lean_dec_ref(v___y_558_);
return v_res_566_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_Frontend_mkHypImplicationRule_x3f_spec__0(lean_object* v_00_u03b1_567_, lean_object* v_mvarId_568_, lean_object* v_x_569_, lean_object* v___y_570_, lean_object* v___y_571_, lean_object* v___y_572_, lean_object* v___y_573_, lean_object* v___y_574_, lean_object* v___y_575_, lean_object* v___y_576_){
_start:
{
lean_object* v___x_578_; 
v___x_578_ = lp_aesop_Lean_MVarId_withContext___at___00Aesop_Frontend_mkHypImplicationRule_x3f_spec__0___redArg(v_mvarId_568_, v_x_569_, v___y_570_, v___y_571_, v___y_572_, v___y_573_, v___y_574_, v___y_575_, v___y_576_);
return v___x_578_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_Frontend_mkHypImplicationRule_x3f_spec__0___boxed(lean_object* v_00_u03b1_579_, lean_object* v_mvarId_580_, lean_object* v_x_581_, lean_object* v___y_582_, lean_object* v___y_583_, lean_object* v___y_584_, lean_object* v___y_585_, lean_object* v___y_586_, lean_object* v___y_587_, lean_object* v___y_588_, lean_object* v___y_589_){
_start:
{
lean_object* v_res_590_; 
v_res_590_ = lp_aesop_Lean_MVarId_withContext___at___00Aesop_Frontend_mkHypImplicationRule_x3f_spec__0(v_00_u03b1_579_, v_mvarId_580_, v_x_581_, v___y_582_, v___y_583_, v___y_584_, v___y_585_, v___y_586_, v___y_587_, v___y_588_);
lean_dec(v___y_588_);
lean_dec_ref(v___y_587_);
lean_dec(v___y_586_);
lean_dec_ref(v___y_585_);
lean_dec(v___y_584_);
lean_dec_ref(v___y_583_);
lean_dec_ref(v___y_582_);
return v_res_590_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_mkHypImplicationRule_x3f___lam__0(uint8_t v___x_591_, lean_object* v_fvarId_592_, lean_object* v___y_593_, lean_object* v___y_594_, lean_object* v___y_595_, lean_object* v___y_596_, lean_object* v___y_597_, lean_object* v___y_598_, lean_object* v___y_599_){
_start:
{
lean_object* v_keyedConfig_601_; uint8_t v_trackZetaDelta_602_; lean_object* v_zetaDeltaSet_603_; lean_object* v_lctx_604_; lean_object* v_localInstances_605_; lean_object* v_defEqCtx_x3f_606_; lean_object* v_synthPendingDepth_607_; lean_object* v_customCanUnfoldPredicate_x3f_608_; uint8_t v_univApprox_609_; uint8_t v_inTypeClassResolution_610_; uint8_t v_cacheInferType_611_; lean_object* v___x_612_; lean_object* v___x_613_; lean_object* v___x_614_; 
v_keyedConfig_601_ = lean_ctor_get(v___y_596_, 0);
v_trackZetaDelta_602_ = lean_ctor_get_uint8(v___y_596_, sizeof(void*)*7);
v_zetaDeltaSet_603_ = lean_ctor_get(v___y_596_, 1);
v_lctx_604_ = lean_ctor_get(v___y_596_, 2);
v_localInstances_605_ = lean_ctor_get(v___y_596_, 3);
v_defEqCtx_x3f_606_ = lean_ctor_get(v___y_596_, 4);
v_synthPendingDepth_607_ = lean_ctor_get(v___y_596_, 5);
v_customCanUnfoldPredicate_x3f_608_ = lean_ctor_get(v___y_596_, 6);
v_univApprox_609_ = lean_ctor_get_uint8(v___y_596_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_610_ = lean_ctor_get_uint8(v___y_596_, sizeof(void*)*7 + 2);
v_cacheInferType_611_ = lean_ctor_get_uint8(v___y_596_, sizeof(void*)*7 + 3);
lean_inc_ref(v_keyedConfig_601_);
v___x_612_ = l_Lean_Meta_ConfigWithKey_setTransparency(v___x_591_, v_keyedConfig_601_);
lean_inc(v_customCanUnfoldPredicate_x3f_608_);
lean_inc(v_synthPendingDepth_607_);
lean_inc(v_defEqCtx_x3f_606_);
lean_inc_ref(v_localInstances_605_);
lean_inc_ref(v_lctx_604_);
lean_inc(v_zetaDeltaSet_603_);
v___x_613_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v___x_613_, 0, v___x_612_);
lean_ctor_set(v___x_613_, 1, v_zetaDeltaSet_603_);
lean_ctor_set(v___x_613_, 2, v_lctx_604_);
lean_ctor_set(v___x_613_, 3, v_localInstances_605_);
lean_ctor_set(v___x_613_, 4, v_defEqCtx_x3f_606_);
lean_ctor_set(v___x_613_, 5, v_synthPendingDepth_607_);
lean_ctor_set(v___x_613_, 6, v_customCanUnfoldPredicate_x3f_608_);
lean_ctor_set_uint8(v___x_613_, sizeof(void*)*7, v_trackZetaDelta_602_);
lean_ctor_set_uint8(v___x_613_, sizeof(void*)*7 + 1, v_univApprox_609_);
lean_ctor_set_uint8(v___x_613_, sizeof(void*)*7 + 2, v_inTypeClassResolution_610_);
lean_ctor_set_uint8(v___x_613_, sizeof(void*)*7 + 3, v_cacheInferType_611_);
lean_inc(v_fvarId_592_);
v___x_614_ = l_Lean_FVarId_getType___redArg(v_fvarId_592_, v___x_613_, v___y_598_, v___y_599_);
if (lean_obj_tag(v___x_614_) == 0)
{
lean_object* v_a_615_; lean_object* v___x_616_; 
v_a_615_ = lean_ctor_get(v___x_614_, 0);
lean_inc(v_a_615_);
lean_dec_ref_known(v___x_614_, 1);
v___x_616_ = lp_aesop_Aesop_Frontend_isImplication(v_a_615_, v___x_613_, v___y_597_, v___y_598_, v___y_599_);
if (lean_obj_tag(v___x_616_) == 0)
{
lean_object* v_a_617_; lean_object* v___x_619_; uint8_t v_isShared_620_; uint8_t v_isSharedCheck_644_; 
v_a_617_ = lean_ctor_get(v___x_616_, 0);
v_isSharedCheck_644_ = !lean_is_exclusive(v___x_616_);
if (v_isSharedCheck_644_ == 0)
{
v___x_619_ = v___x_616_;
v_isShared_620_ = v_isSharedCheck_644_;
goto v_resetjp_618_;
}
else
{
lean_inc(v_a_617_);
lean_dec(v___x_616_);
v___x_619_ = lean_box(0);
v_isShared_620_ = v_isSharedCheck_644_;
goto v_resetjp_618_;
}
v_resetjp_618_:
{
uint8_t v___x_621_; 
v___x_621_ = lean_unbox(v_a_617_);
lean_dec(v_a_617_);
if (v___x_621_ == 0)
{
lean_object* v___x_622_; lean_object* v___x_624_; 
lean_dec_ref_known(v___x_613_, 7);
lean_dec(v_fvarId_592_);
v___x_622_ = lean_box(0);
if (v_isShared_620_ == 0)
{
lean_ctor_set(v___x_619_, 0, v___x_622_);
v___x_624_ = v___x_619_;
goto v_reusejp_623_;
}
else
{
lean_object* v_reuseFailAlloc_625_; 
v_reuseFailAlloc_625_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_625_, 0, v___x_622_);
v___x_624_ = v_reuseFailAlloc_625_;
goto v_reusejp_623_;
}
v_reusejp_623_:
{
return v___x_624_;
}
}
else
{
lean_object* v___x_626_; 
lean_del_object(v___x_619_);
v___x_626_ = lp_aesop_Aesop_Frontend_mkHypForwardRule(v_fvarId_592_, v___y_593_, v___y_594_, v___y_595_, v___x_613_, v___y_597_, v___y_598_, v___y_599_);
lean_dec_ref_known(v___x_613_, 7);
if (lean_obj_tag(v___x_626_) == 0)
{
lean_object* v_a_627_; lean_object* v___x_629_; uint8_t v_isShared_630_; uint8_t v_isSharedCheck_635_; 
v_a_627_ = lean_ctor_get(v___x_626_, 0);
v_isSharedCheck_635_ = !lean_is_exclusive(v___x_626_);
if (v_isSharedCheck_635_ == 0)
{
v___x_629_ = v___x_626_;
v_isShared_630_ = v_isSharedCheck_635_;
goto v_resetjp_628_;
}
else
{
lean_inc(v_a_627_);
lean_dec(v___x_626_);
v___x_629_ = lean_box(0);
v_isShared_630_ = v_isSharedCheck_635_;
goto v_resetjp_628_;
}
v_resetjp_628_:
{
lean_object* v___x_631_; lean_object* v___x_633_; 
v___x_631_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_631_, 0, v_a_627_);
if (v_isShared_630_ == 0)
{
lean_ctor_set(v___x_629_, 0, v___x_631_);
v___x_633_ = v___x_629_;
goto v_reusejp_632_;
}
else
{
lean_object* v_reuseFailAlloc_634_; 
v_reuseFailAlloc_634_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_634_, 0, v___x_631_);
v___x_633_ = v_reuseFailAlloc_634_;
goto v_reusejp_632_;
}
v_reusejp_632_:
{
return v___x_633_;
}
}
}
else
{
lean_object* v_a_636_; lean_object* v___x_638_; uint8_t v_isShared_639_; uint8_t v_isSharedCheck_643_; 
v_a_636_ = lean_ctor_get(v___x_626_, 0);
v_isSharedCheck_643_ = !lean_is_exclusive(v___x_626_);
if (v_isSharedCheck_643_ == 0)
{
v___x_638_ = v___x_626_;
v_isShared_639_ = v_isSharedCheck_643_;
goto v_resetjp_637_;
}
else
{
lean_inc(v_a_636_);
lean_dec(v___x_626_);
v___x_638_ = lean_box(0);
v_isShared_639_ = v_isSharedCheck_643_;
goto v_resetjp_637_;
}
v_resetjp_637_:
{
lean_object* v___x_641_; 
if (v_isShared_639_ == 0)
{
v___x_641_ = v___x_638_;
goto v_reusejp_640_;
}
else
{
lean_object* v_reuseFailAlloc_642_; 
v_reuseFailAlloc_642_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_642_, 0, v_a_636_);
v___x_641_ = v_reuseFailAlloc_642_;
goto v_reusejp_640_;
}
v_reusejp_640_:
{
return v___x_641_;
}
}
}
}
}
}
else
{
lean_object* v_a_645_; lean_object* v___x_647_; uint8_t v_isShared_648_; uint8_t v_isSharedCheck_652_; 
lean_dec_ref_known(v___x_613_, 7);
lean_dec(v_fvarId_592_);
v_a_645_ = lean_ctor_get(v___x_616_, 0);
v_isSharedCheck_652_ = !lean_is_exclusive(v___x_616_);
if (v_isSharedCheck_652_ == 0)
{
v___x_647_ = v___x_616_;
v_isShared_648_ = v_isSharedCheck_652_;
goto v_resetjp_646_;
}
else
{
lean_inc(v_a_645_);
lean_dec(v___x_616_);
v___x_647_ = lean_box(0);
v_isShared_648_ = v_isSharedCheck_652_;
goto v_resetjp_646_;
}
v_resetjp_646_:
{
lean_object* v___x_650_; 
if (v_isShared_648_ == 0)
{
v___x_650_ = v___x_647_;
goto v_reusejp_649_;
}
else
{
lean_object* v_reuseFailAlloc_651_; 
v_reuseFailAlloc_651_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_651_, 0, v_a_645_);
v___x_650_ = v_reuseFailAlloc_651_;
goto v_reusejp_649_;
}
v_reusejp_649_:
{
return v___x_650_;
}
}
}
}
else
{
lean_object* v_a_653_; lean_object* v___x_655_; uint8_t v_isShared_656_; uint8_t v_isSharedCheck_660_; 
lean_dec_ref_known(v___x_613_, 7);
lean_dec(v_fvarId_592_);
v_a_653_ = lean_ctor_get(v___x_614_, 0);
v_isSharedCheck_660_ = !lean_is_exclusive(v___x_614_);
if (v_isSharedCheck_660_ == 0)
{
v___x_655_ = v___x_614_;
v_isShared_656_ = v_isSharedCheck_660_;
goto v_resetjp_654_;
}
else
{
lean_inc(v_a_653_);
lean_dec(v___x_614_);
v___x_655_ = lean_box(0);
v_isShared_656_ = v_isSharedCheck_660_;
goto v_resetjp_654_;
}
v_resetjp_654_:
{
lean_object* v___x_658_; 
if (v_isShared_656_ == 0)
{
v___x_658_ = v___x_655_;
goto v_reusejp_657_;
}
else
{
lean_object* v_reuseFailAlloc_659_; 
v_reuseFailAlloc_659_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_659_, 0, v_a_653_);
v___x_658_ = v_reuseFailAlloc_659_;
goto v_reusejp_657_;
}
v_reusejp_657_:
{
return v___x_658_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_mkHypImplicationRule_x3f___lam__0___boxed(lean_object* v___x_661_, lean_object* v_fvarId_662_, lean_object* v___y_663_, lean_object* v___y_664_, lean_object* v___y_665_, lean_object* v___y_666_, lean_object* v___y_667_, lean_object* v___y_668_, lean_object* v___y_669_, lean_object* v___y_670_){
_start:
{
uint8_t v___x_4064__boxed_671_; lean_object* v_res_672_; 
v___x_4064__boxed_671_ = lean_unbox(v___x_661_);
v_res_672_ = lp_aesop_Aesop_Frontend_mkHypImplicationRule_x3f___lam__0(v___x_4064__boxed_671_, v_fvarId_662_, v___y_663_, v___y_664_, v___y_665_, v___y_666_, v___y_667_, v___y_668_, v___y_669_);
lean_dec(v___y_669_);
lean_dec_ref(v___y_668_);
lean_dec(v___y_667_);
lean_dec_ref(v___y_666_);
lean_dec(v___y_665_);
lean_dec_ref(v___y_664_);
lean_dec_ref(v___y_663_);
return v_res_672_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_mkHypImplicationRule_x3f(lean_object* v_fvarId_673_, lean_object* v_a_674_, lean_object* v_a_675_, lean_object* v_a_676_, lean_object* v_a_677_, lean_object* v_a_678_, lean_object* v_a_679_, lean_object* v_a_680_){
_start:
{
lean_object* v_goal_682_; uint8_t v___x_683_; lean_object* v___x_684_; lean_object* v___f_685_; lean_object* v___x_686_; 
v_goal_682_ = lean_ctor_get(v_a_674_, 0);
v___x_683_ = 2;
v___x_684_ = lean_box(v___x_683_);
v___f_685_ = lean_alloc_closure((void*)(lp_aesop_Aesop_Frontend_mkHypImplicationRule_x3f___lam__0___boxed), 10, 2);
lean_closure_set(v___f_685_, 0, v___x_684_);
lean_closure_set(v___f_685_, 1, v_fvarId_673_);
lean_inc(v_goal_682_);
v___x_686_ = lp_aesop_Lean_MVarId_withContext___at___00Aesop_Frontend_mkHypImplicationRule_x3f_spec__0___redArg(v_goal_682_, v___f_685_, v_a_674_, v_a_675_, v_a_676_, v_a_677_, v_a_678_, v_a_679_, v_a_680_);
return v___x_686_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_mkHypImplicationRule_x3f___boxed(lean_object* v_fvarId_687_, lean_object* v_a_688_, lean_object* v_a_689_, lean_object* v_a_690_, lean_object* v_a_691_, lean_object* v_a_692_, lean_object* v_a_693_, lean_object* v_a_694_, lean_object* v_a_695_){
_start:
{
lean_object* v_res_696_; 
v_res_696_ = lp_aesop_Aesop_Frontend_mkHypImplicationRule_x3f(v_fvarId_687_, v_a_688_, v_a_689_, v_a_690_, v_a_691_, v_a_692_, v_a_693_, v_a_694_);
lean_dec(v_a_694_);
lean_dec_ref(v_a_693_);
lean_dec(v_a_692_);
lean_dec_ref(v_a_691_);
lean_dec(v_a_690_);
lean_dec_ref(v_a_689_);
lean_dec_ref(v_a_688_);
return v_res_696_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_Frontend_addLocalImplications_spec__0_spec__0_spec__2_spec__3(lean_object* v_as_697_, size_t v_sz_698_, size_t v_i_699_, lean_object* v_b_700_, lean_object* v___y_701_, lean_object* v___y_702_, lean_object* v___y_703_, lean_object* v___y_704_, lean_object* v___y_705_, lean_object* v___y_706_, lean_object* v___y_707_){
_start:
{
uint8_t v___x_709_; 
v___x_709_ = lean_usize_dec_lt(v_i_699_, v_sz_698_);
if (v___x_709_ == 0)
{
lean_object* v___x_710_; 
v___x_710_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_710_, 0, v_b_700_);
return v___x_710_;
}
else
{
lean_object* v_snd_711_; lean_object* v___x_713_; uint8_t v_isShared_714_; uint8_t v_isSharedCheck_740_; 
v_snd_711_ = lean_ctor_get(v_b_700_, 1);
v_isSharedCheck_740_ = !lean_is_exclusive(v_b_700_);
if (v_isSharedCheck_740_ == 0)
{
lean_object* v_unused_741_; 
v_unused_741_ = lean_ctor_get(v_b_700_, 0);
lean_dec(v_unused_741_);
v___x_713_ = v_b_700_;
v_isShared_714_ = v_isSharedCheck_740_;
goto v_resetjp_712_;
}
else
{
lean_inc(v_snd_711_);
lean_dec(v_b_700_);
v___x_713_ = lean_box(0);
v_isShared_714_ = v_isSharedCheck_740_;
goto v_resetjp_712_;
}
v_resetjp_712_:
{
lean_object* v___x_715_; lean_object* v_a_717_; lean_object* v_a_724_; 
v___x_715_ = lean_box(0);
v_a_724_ = lean_array_uget_borrowed(v_as_697_, v_i_699_);
if (lean_obj_tag(v_a_724_) == 0)
{
v_a_717_ = v_snd_711_;
goto v___jp_716_;
}
else
{
lean_object* v_val_725_; uint8_t v___x_726_; 
v_val_725_ = lean_ctor_get(v_a_724_, 0);
v___x_726_ = l_Lean_LocalDecl_isImplementationDetail(v_val_725_);
if (v___x_726_ == 0)
{
lean_object* v___x_727_; lean_object* v___x_728_; 
v___x_727_ = l_Lean_LocalDecl_fvarId(v_val_725_);
v___x_728_ = lp_aesop_Aesop_Frontend_mkHypImplicationRule_x3f(v___x_727_, v___y_701_, v___y_702_, v___y_703_, v___y_704_, v___y_705_, v___y_706_, v___y_707_);
if (lean_obj_tag(v___x_728_) == 0)
{
lean_object* v_a_729_; 
v_a_729_ = lean_ctor_get(v___x_728_, 0);
lean_inc(v_a_729_);
lean_dec_ref_known(v___x_728_, 1);
if (lean_obj_tag(v_a_729_) == 1)
{
lean_object* v_val_730_; lean_object* v___x_731_; 
v_val_730_ = lean_ctor_get(v_a_729_, 0);
lean_inc(v_val_730_);
lean_dec_ref_known(v_a_729_, 1);
v___x_731_ = lp_aesop_Aesop_LocalRuleSet_add(v_snd_711_, v_val_730_);
v_a_717_ = v___x_731_;
goto v___jp_716_;
}
else
{
lean_dec(v_a_729_);
v_a_717_ = v_snd_711_;
goto v___jp_716_;
}
}
else
{
lean_object* v_a_732_; lean_object* v___x_734_; uint8_t v_isShared_735_; uint8_t v_isSharedCheck_739_; 
lean_del_object(v___x_713_);
lean_dec(v_snd_711_);
v_a_732_ = lean_ctor_get(v___x_728_, 0);
v_isSharedCheck_739_ = !lean_is_exclusive(v___x_728_);
if (v_isSharedCheck_739_ == 0)
{
v___x_734_ = v___x_728_;
v_isShared_735_ = v_isSharedCheck_739_;
goto v_resetjp_733_;
}
else
{
lean_inc(v_a_732_);
lean_dec(v___x_728_);
v___x_734_ = lean_box(0);
v_isShared_735_ = v_isSharedCheck_739_;
goto v_resetjp_733_;
}
v_resetjp_733_:
{
lean_object* v___x_737_; 
if (v_isShared_735_ == 0)
{
v___x_737_ = v___x_734_;
goto v_reusejp_736_;
}
else
{
lean_object* v_reuseFailAlloc_738_; 
v_reuseFailAlloc_738_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_738_, 0, v_a_732_);
v___x_737_ = v_reuseFailAlloc_738_;
goto v_reusejp_736_;
}
v_reusejp_736_:
{
return v___x_737_;
}
}
}
}
else
{
v_a_717_ = v_snd_711_;
goto v___jp_716_;
}
}
v___jp_716_:
{
lean_object* v___x_719_; 
if (v_isShared_714_ == 0)
{
lean_ctor_set(v___x_713_, 1, v_a_717_);
lean_ctor_set(v___x_713_, 0, v___x_715_);
v___x_719_ = v___x_713_;
goto v_reusejp_718_;
}
else
{
lean_object* v_reuseFailAlloc_723_; 
v_reuseFailAlloc_723_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_723_, 0, v___x_715_);
lean_ctor_set(v_reuseFailAlloc_723_, 1, v_a_717_);
v___x_719_ = v_reuseFailAlloc_723_;
goto v_reusejp_718_;
}
v_reusejp_718_:
{
size_t v___x_720_; size_t v___x_721_; 
v___x_720_ = ((size_t)1ULL);
v___x_721_ = lean_usize_add(v_i_699_, v___x_720_);
v_i_699_ = v___x_721_;
v_b_700_ = v___x_719_;
goto _start;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_Frontend_addLocalImplications_spec__0_spec__0_spec__2_spec__3___boxed(lean_object* v_as_742_, lean_object* v_sz_743_, lean_object* v_i_744_, lean_object* v_b_745_, lean_object* v___y_746_, lean_object* v___y_747_, lean_object* v___y_748_, lean_object* v___y_749_, lean_object* v___y_750_, lean_object* v___y_751_, lean_object* v___y_752_, lean_object* v___y_753_){
_start:
{
size_t v_sz_boxed_754_; size_t v_i_boxed_755_; lean_object* v_res_756_; 
v_sz_boxed_754_ = lean_unbox_usize(v_sz_743_);
lean_dec(v_sz_743_);
v_i_boxed_755_ = lean_unbox_usize(v_i_744_);
lean_dec(v_i_744_);
v_res_756_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_Frontend_addLocalImplications_spec__0_spec__0_spec__2_spec__3(v_as_742_, v_sz_boxed_754_, v_i_boxed_755_, v_b_745_, v___y_746_, v___y_747_, v___y_748_, v___y_749_, v___y_750_, v___y_751_, v___y_752_);
lean_dec(v___y_752_);
lean_dec_ref(v___y_751_);
lean_dec(v___y_750_);
lean_dec_ref(v___y_749_);
lean_dec(v___y_748_);
lean_dec_ref(v___y_747_);
lean_dec_ref(v___y_746_);
lean_dec_ref(v_as_742_);
return v_res_756_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_Frontend_addLocalImplications_spec__0_spec__0_spec__2(lean_object* v_as_757_, size_t v_sz_758_, size_t v_i_759_, lean_object* v_b_760_, lean_object* v___y_761_, lean_object* v___y_762_, lean_object* v___y_763_, lean_object* v___y_764_, lean_object* v___y_765_, lean_object* v___y_766_, lean_object* v___y_767_){
_start:
{
uint8_t v___x_769_; 
v___x_769_ = lean_usize_dec_lt(v_i_759_, v_sz_758_);
if (v___x_769_ == 0)
{
lean_object* v___x_770_; 
v___x_770_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_770_, 0, v_b_760_);
return v___x_770_;
}
else
{
lean_object* v_snd_771_; lean_object* v___x_773_; uint8_t v_isShared_774_; uint8_t v_isSharedCheck_800_; 
v_snd_771_ = lean_ctor_get(v_b_760_, 1);
v_isSharedCheck_800_ = !lean_is_exclusive(v_b_760_);
if (v_isSharedCheck_800_ == 0)
{
lean_object* v_unused_801_; 
v_unused_801_ = lean_ctor_get(v_b_760_, 0);
lean_dec(v_unused_801_);
v___x_773_ = v_b_760_;
v_isShared_774_ = v_isSharedCheck_800_;
goto v_resetjp_772_;
}
else
{
lean_inc(v_snd_771_);
lean_dec(v_b_760_);
v___x_773_ = lean_box(0);
v_isShared_774_ = v_isSharedCheck_800_;
goto v_resetjp_772_;
}
v_resetjp_772_:
{
lean_object* v___x_775_; lean_object* v_a_777_; lean_object* v_a_784_; 
v___x_775_ = lean_box(0);
v_a_784_ = lean_array_uget_borrowed(v_as_757_, v_i_759_);
if (lean_obj_tag(v_a_784_) == 0)
{
v_a_777_ = v_snd_771_;
goto v___jp_776_;
}
else
{
lean_object* v_val_785_; uint8_t v___x_786_; 
v_val_785_ = lean_ctor_get(v_a_784_, 0);
v___x_786_ = l_Lean_LocalDecl_isImplementationDetail(v_val_785_);
if (v___x_786_ == 0)
{
lean_object* v___x_787_; lean_object* v___x_788_; 
v___x_787_ = l_Lean_LocalDecl_fvarId(v_val_785_);
v___x_788_ = lp_aesop_Aesop_Frontend_mkHypImplicationRule_x3f(v___x_787_, v___y_761_, v___y_762_, v___y_763_, v___y_764_, v___y_765_, v___y_766_, v___y_767_);
if (lean_obj_tag(v___x_788_) == 0)
{
lean_object* v_a_789_; 
v_a_789_ = lean_ctor_get(v___x_788_, 0);
lean_inc(v_a_789_);
lean_dec_ref_known(v___x_788_, 1);
if (lean_obj_tag(v_a_789_) == 1)
{
lean_object* v_val_790_; lean_object* v___x_791_; 
v_val_790_ = lean_ctor_get(v_a_789_, 0);
lean_inc(v_val_790_);
lean_dec_ref_known(v_a_789_, 1);
v___x_791_ = lp_aesop_Aesop_LocalRuleSet_add(v_snd_771_, v_val_790_);
v_a_777_ = v___x_791_;
goto v___jp_776_;
}
else
{
lean_dec(v_a_789_);
v_a_777_ = v_snd_771_;
goto v___jp_776_;
}
}
else
{
lean_object* v_a_792_; lean_object* v___x_794_; uint8_t v_isShared_795_; uint8_t v_isSharedCheck_799_; 
lean_del_object(v___x_773_);
lean_dec(v_snd_771_);
v_a_792_ = lean_ctor_get(v___x_788_, 0);
v_isSharedCheck_799_ = !lean_is_exclusive(v___x_788_);
if (v_isSharedCheck_799_ == 0)
{
v___x_794_ = v___x_788_;
v_isShared_795_ = v_isSharedCheck_799_;
goto v_resetjp_793_;
}
else
{
lean_inc(v_a_792_);
lean_dec(v___x_788_);
v___x_794_ = lean_box(0);
v_isShared_795_ = v_isSharedCheck_799_;
goto v_resetjp_793_;
}
v_resetjp_793_:
{
lean_object* v___x_797_; 
if (v_isShared_795_ == 0)
{
v___x_797_ = v___x_794_;
goto v_reusejp_796_;
}
else
{
lean_object* v_reuseFailAlloc_798_; 
v_reuseFailAlloc_798_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_798_, 0, v_a_792_);
v___x_797_ = v_reuseFailAlloc_798_;
goto v_reusejp_796_;
}
v_reusejp_796_:
{
return v___x_797_;
}
}
}
}
else
{
v_a_777_ = v_snd_771_;
goto v___jp_776_;
}
}
v___jp_776_:
{
lean_object* v___x_779_; 
if (v_isShared_774_ == 0)
{
lean_ctor_set(v___x_773_, 1, v_a_777_);
lean_ctor_set(v___x_773_, 0, v___x_775_);
v___x_779_ = v___x_773_;
goto v_reusejp_778_;
}
else
{
lean_object* v_reuseFailAlloc_783_; 
v_reuseFailAlloc_783_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_783_, 0, v___x_775_);
lean_ctor_set(v_reuseFailAlloc_783_, 1, v_a_777_);
v___x_779_ = v_reuseFailAlloc_783_;
goto v_reusejp_778_;
}
v_reusejp_778_:
{
size_t v___x_780_; size_t v___x_781_; lean_object* v___x_782_; 
v___x_780_ = ((size_t)1ULL);
v___x_781_ = lean_usize_add(v_i_759_, v___x_780_);
v___x_782_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_Frontend_addLocalImplications_spec__0_spec__0_spec__2_spec__3(v_as_757_, v_sz_758_, v___x_781_, v___x_779_, v___y_761_, v___y_762_, v___y_763_, v___y_764_, v___y_765_, v___y_766_, v___y_767_);
return v___x_782_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_Frontend_addLocalImplications_spec__0_spec__0_spec__2___boxed(lean_object* v_as_802_, lean_object* v_sz_803_, lean_object* v_i_804_, lean_object* v_b_805_, lean_object* v___y_806_, lean_object* v___y_807_, lean_object* v___y_808_, lean_object* v___y_809_, lean_object* v___y_810_, lean_object* v___y_811_, lean_object* v___y_812_, lean_object* v___y_813_){
_start:
{
size_t v_sz_boxed_814_; size_t v_i_boxed_815_; lean_object* v_res_816_; 
v_sz_boxed_814_ = lean_unbox_usize(v_sz_803_);
lean_dec(v_sz_803_);
v_i_boxed_815_ = lean_unbox_usize(v_i_804_);
lean_dec(v_i_804_);
v_res_816_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_Frontend_addLocalImplications_spec__0_spec__0_spec__2(v_as_802_, v_sz_boxed_814_, v_i_boxed_815_, v_b_805_, v___y_806_, v___y_807_, v___y_808_, v___y_809_, v___y_810_, v___y_811_, v___y_812_);
lean_dec(v___y_812_);
lean_dec_ref(v___y_811_);
lean_dec(v___y_810_);
lean_dec_ref(v___y_809_);
lean_dec(v___y_808_);
lean_dec_ref(v___y_807_);
lean_dec_ref(v___y_806_);
lean_dec_ref(v_as_802_);
return v_res_816_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_Frontend_addLocalImplications_spec__0_spec__0(lean_object* v_init_817_, lean_object* v_n_818_, lean_object* v_b_819_, lean_object* v___y_820_, lean_object* v___y_821_, lean_object* v___y_822_, lean_object* v___y_823_, lean_object* v___y_824_, lean_object* v___y_825_, lean_object* v___y_826_){
_start:
{
if (lean_obj_tag(v_n_818_) == 0)
{
lean_object* v_cs_828_; lean_object* v___x_829_; lean_object* v___x_830_; size_t v_sz_831_; size_t v___x_832_; lean_object* v___x_833_; 
v_cs_828_ = lean_ctor_get(v_n_818_, 0);
v___x_829_ = lean_box(0);
v___x_830_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_830_, 0, v___x_829_);
lean_ctor_set(v___x_830_, 1, v_b_819_);
v_sz_831_ = lean_array_size(v_cs_828_);
v___x_832_ = ((size_t)0ULL);
v___x_833_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_Frontend_addLocalImplications_spec__0_spec__0_spec__1(v_init_817_, v_cs_828_, v_sz_831_, v___x_832_, v___x_830_, v___y_820_, v___y_821_, v___y_822_, v___y_823_, v___y_824_, v___y_825_, v___y_826_);
if (lean_obj_tag(v___x_833_) == 0)
{
lean_object* v_a_834_; lean_object* v___x_836_; uint8_t v_isShared_837_; uint8_t v_isSharedCheck_848_; 
v_a_834_ = lean_ctor_get(v___x_833_, 0);
v_isSharedCheck_848_ = !lean_is_exclusive(v___x_833_);
if (v_isSharedCheck_848_ == 0)
{
v___x_836_ = v___x_833_;
v_isShared_837_ = v_isSharedCheck_848_;
goto v_resetjp_835_;
}
else
{
lean_inc(v_a_834_);
lean_dec(v___x_833_);
v___x_836_ = lean_box(0);
v_isShared_837_ = v_isSharedCheck_848_;
goto v_resetjp_835_;
}
v_resetjp_835_:
{
lean_object* v_fst_838_; 
v_fst_838_ = lean_ctor_get(v_a_834_, 0);
if (lean_obj_tag(v_fst_838_) == 0)
{
lean_object* v_snd_839_; lean_object* v___x_840_; lean_object* v___x_842_; 
v_snd_839_ = lean_ctor_get(v_a_834_, 1);
lean_inc(v_snd_839_);
lean_dec(v_a_834_);
v___x_840_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_840_, 0, v_snd_839_);
if (v_isShared_837_ == 0)
{
lean_ctor_set(v___x_836_, 0, v___x_840_);
v___x_842_ = v___x_836_;
goto v_reusejp_841_;
}
else
{
lean_object* v_reuseFailAlloc_843_; 
v_reuseFailAlloc_843_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_843_, 0, v___x_840_);
v___x_842_ = v_reuseFailAlloc_843_;
goto v_reusejp_841_;
}
v_reusejp_841_:
{
return v___x_842_;
}
}
else
{
lean_object* v_val_844_; lean_object* v___x_846_; 
lean_inc_ref(v_fst_838_);
lean_dec(v_a_834_);
v_val_844_ = lean_ctor_get(v_fst_838_, 0);
lean_inc(v_val_844_);
lean_dec_ref_known(v_fst_838_, 1);
if (v_isShared_837_ == 0)
{
lean_ctor_set(v___x_836_, 0, v_val_844_);
v___x_846_ = v___x_836_;
goto v_reusejp_845_;
}
else
{
lean_object* v_reuseFailAlloc_847_; 
v_reuseFailAlloc_847_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_847_, 0, v_val_844_);
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
v_a_849_ = lean_ctor_get(v___x_833_, 0);
v_isSharedCheck_856_ = !lean_is_exclusive(v___x_833_);
if (v_isSharedCheck_856_ == 0)
{
v___x_851_ = v___x_833_;
v_isShared_852_ = v_isSharedCheck_856_;
goto v_resetjp_850_;
}
else
{
lean_inc(v_a_849_);
lean_dec(v___x_833_);
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
lean_object* v_vs_857_; lean_object* v___x_858_; lean_object* v___x_859_; size_t v_sz_860_; size_t v___x_861_; lean_object* v___x_862_; 
v_vs_857_ = lean_ctor_get(v_n_818_, 0);
v___x_858_ = lean_box(0);
v___x_859_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_859_, 0, v___x_858_);
lean_ctor_set(v___x_859_, 1, v_b_819_);
v_sz_860_ = lean_array_size(v_vs_857_);
v___x_861_ = ((size_t)0ULL);
v___x_862_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_Frontend_addLocalImplications_spec__0_spec__0_spec__2(v_vs_857_, v_sz_860_, v___x_861_, v___x_859_, v___y_820_, v___y_821_, v___y_822_, v___y_823_, v___y_824_, v___y_825_, v___y_826_);
if (lean_obj_tag(v___x_862_) == 0)
{
lean_object* v_a_863_; lean_object* v___x_865_; uint8_t v_isShared_866_; uint8_t v_isSharedCheck_877_; 
v_a_863_ = lean_ctor_get(v___x_862_, 0);
v_isSharedCheck_877_ = !lean_is_exclusive(v___x_862_);
if (v_isSharedCheck_877_ == 0)
{
v___x_865_ = v___x_862_;
v_isShared_866_ = v_isSharedCheck_877_;
goto v_resetjp_864_;
}
else
{
lean_inc(v_a_863_);
lean_dec(v___x_862_);
v___x_865_ = lean_box(0);
v_isShared_866_ = v_isSharedCheck_877_;
goto v_resetjp_864_;
}
v_resetjp_864_:
{
lean_object* v_fst_867_; 
v_fst_867_ = lean_ctor_get(v_a_863_, 0);
if (lean_obj_tag(v_fst_867_) == 0)
{
lean_object* v_snd_868_; lean_object* v___x_869_; lean_object* v___x_871_; 
v_snd_868_ = lean_ctor_get(v_a_863_, 1);
lean_inc(v_snd_868_);
lean_dec(v_a_863_);
v___x_869_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_869_, 0, v_snd_868_);
if (v_isShared_866_ == 0)
{
lean_ctor_set(v___x_865_, 0, v___x_869_);
v___x_871_ = v___x_865_;
goto v_reusejp_870_;
}
else
{
lean_object* v_reuseFailAlloc_872_; 
v_reuseFailAlloc_872_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_872_, 0, v___x_869_);
v___x_871_ = v_reuseFailAlloc_872_;
goto v_reusejp_870_;
}
v_reusejp_870_:
{
return v___x_871_;
}
}
else
{
lean_object* v_val_873_; lean_object* v___x_875_; 
lean_inc_ref(v_fst_867_);
lean_dec(v_a_863_);
v_val_873_ = lean_ctor_get(v_fst_867_, 0);
lean_inc(v_val_873_);
lean_dec_ref_known(v_fst_867_, 1);
if (v_isShared_866_ == 0)
{
lean_ctor_set(v___x_865_, 0, v_val_873_);
v___x_875_ = v___x_865_;
goto v_reusejp_874_;
}
else
{
lean_object* v_reuseFailAlloc_876_; 
v_reuseFailAlloc_876_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_876_, 0, v_val_873_);
v___x_875_ = v_reuseFailAlloc_876_;
goto v_reusejp_874_;
}
v_reusejp_874_:
{
return v___x_875_;
}
}
}
}
else
{
lean_object* v_a_878_; lean_object* v___x_880_; uint8_t v_isShared_881_; uint8_t v_isSharedCheck_885_; 
v_a_878_ = lean_ctor_get(v___x_862_, 0);
v_isSharedCheck_885_ = !lean_is_exclusive(v___x_862_);
if (v_isSharedCheck_885_ == 0)
{
v___x_880_ = v___x_862_;
v_isShared_881_ = v_isSharedCheck_885_;
goto v_resetjp_879_;
}
else
{
lean_inc(v_a_878_);
lean_dec(v___x_862_);
v___x_880_ = lean_box(0);
v_isShared_881_ = v_isSharedCheck_885_;
goto v_resetjp_879_;
}
v_resetjp_879_:
{
lean_object* v___x_883_; 
if (v_isShared_881_ == 0)
{
v___x_883_ = v___x_880_;
goto v_reusejp_882_;
}
else
{
lean_object* v_reuseFailAlloc_884_; 
v_reuseFailAlloc_884_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_884_, 0, v_a_878_);
v___x_883_ = v_reuseFailAlloc_884_;
goto v_reusejp_882_;
}
v_reusejp_882_:
{
return v___x_883_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_Frontend_addLocalImplications_spec__0_spec__0_spec__1(lean_object* v_init_886_, lean_object* v_as_887_, size_t v_sz_888_, size_t v_i_889_, lean_object* v_b_890_, lean_object* v___y_891_, lean_object* v___y_892_, lean_object* v___y_893_, lean_object* v___y_894_, lean_object* v___y_895_, lean_object* v___y_896_, lean_object* v___y_897_){
_start:
{
uint8_t v___x_899_; 
v___x_899_ = lean_usize_dec_lt(v_i_889_, v_sz_888_);
if (v___x_899_ == 0)
{
lean_object* v___x_900_; 
v___x_900_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_900_, 0, v_b_890_);
return v___x_900_;
}
else
{
lean_object* v_snd_901_; lean_object* v___x_903_; uint8_t v_isShared_904_; uint8_t v_isSharedCheck_935_; 
v_snd_901_ = lean_ctor_get(v_b_890_, 1);
v_isSharedCheck_935_ = !lean_is_exclusive(v_b_890_);
if (v_isSharedCheck_935_ == 0)
{
lean_object* v_unused_936_; 
v_unused_936_ = lean_ctor_get(v_b_890_, 0);
lean_dec(v_unused_936_);
v___x_903_ = v_b_890_;
v_isShared_904_ = v_isSharedCheck_935_;
goto v_resetjp_902_;
}
else
{
lean_inc(v_snd_901_);
lean_dec(v_b_890_);
v___x_903_ = lean_box(0);
v_isShared_904_ = v_isSharedCheck_935_;
goto v_resetjp_902_;
}
v_resetjp_902_:
{
lean_object* v_a_905_; lean_object* v___x_906_; 
v_a_905_ = lean_array_uget_borrowed(v_as_887_, v_i_889_);
lean_inc(v_snd_901_);
v___x_906_ = lp_aesop_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_Frontend_addLocalImplications_spec__0_spec__0(v_init_886_, v_a_905_, v_snd_901_, v___y_891_, v___y_892_, v___y_893_, v___y_894_, v___y_895_, v___y_896_, v___y_897_);
if (lean_obj_tag(v___x_906_) == 0)
{
lean_object* v_a_907_; lean_object* v___x_909_; uint8_t v_isShared_910_; uint8_t v_isSharedCheck_926_; 
v_a_907_ = lean_ctor_get(v___x_906_, 0);
v_isSharedCheck_926_ = !lean_is_exclusive(v___x_906_);
if (v_isSharedCheck_926_ == 0)
{
v___x_909_ = v___x_906_;
v_isShared_910_ = v_isSharedCheck_926_;
goto v_resetjp_908_;
}
else
{
lean_inc(v_a_907_);
lean_dec(v___x_906_);
v___x_909_ = lean_box(0);
v_isShared_910_ = v_isSharedCheck_926_;
goto v_resetjp_908_;
}
v_resetjp_908_:
{
if (lean_obj_tag(v_a_907_) == 0)
{
lean_object* v___x_911_; lean_object* v___x_913_; 
v___x_911_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_911_, 0, v_a_907_);
if (v_isShared_904_ == 0)
{
lean_ctor_set(v___x_903_, 0, v___x_911_);
v___x_913_ = v___x_903_;
goto v_reusejp_912_;
}
else
{
lean_object* v_reuseFailAlloc_917_; 
v_reuseFailAlloc_917_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_917_, 0, v___x_911_);
lean_ctor_set(v_reuseFailAlloc_917_, 1, v_snd_901_);
v___x_913_ = v_reuseFailAlloc_917_;
goto v_reusejp_912_;
}
v_reusejp_912_:
{
lean_object* v___x_915_; 
if (v_isShared_910_ == 0)
{
lean_ctor_set(v___x_909_, 0, v___x_913_);
v___x_915_ = v___x_909_;
goto v_reusejp_914_;
}
else
{
lean_object* v_reuseFailAlloc_916_; 
v_reuseFailAlloc_916_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_916_, 0, v___x_913_);
v___x_915_ = v_reuseFailAlloc_916_;
goto v_reusejp_914_;
}
v_reusejp_914_:
{
return v___x_915_;
}
}
}
else
{
lean_object* v_a_918_; lean_object* v___x_919_; lean_object* v___x_921_; 
lean_del_object(v___x_909_);
lean_dec(v_snd_901_);
v_a_918_ = lean_ctor_get(v_a_907_, 0);
lean_inc(v_a_918_);
lean_dec_ref_known(v_a_907_, 1);
v___x_919_ = lean_box(0);
if (v_isShared_904_ == 0)
{
lean_ctor_set(v___x_903_, 1, v_a_918_);
lean_ctor_set(v___x_903_, 0, v___x_919_);
v___x_921_ = v___x_903_;
goto v_reusejp_920_;
}
else
{
lean_object* v_reuseFailAlloc_925_; 
v_reuseFailAlloc_925_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_925_, 0, v___x_919_);
lean_ctor_set(v_reuseFailAlloc_925_, 1, v_a_918_);
v___x_921_ = v_reuseFailAlloc_925_;
goto v_reusejp_920_;
}
v_reusejp_920_:
{
size_t v___x_922_; size_t v___x_923_; 
v___x_922_ = ((size_t)1ULL);
v___x_923_ = lean_usize_add(v_i_889_, v___x_922_);
v_i_889_ = v___x_923_;
v_b_890_ = v___x_921_;
goto _start;
}
}
}
}
else
{
lean_object* v_a_927_; lean_object* v___x_929_; uint8_t v_isShared_930_; uint8_t v_isSharedCheck_934_; 
lean_del_object(v___x_903_);
lean_dec(v_snd_901_);
v_a_927_ = lean_ctor_get(v___x_906_, 0);
v_isSharedCheck_934_ = !lean_is_exclusive(v___x_906_);
if (v_isSharedCheck_934_ == 0)
{
v___x_929_ = v___x_906_;
v_isShared_930_ = v_isSharedCheck_934_;
goto v_resetjp_928_;
}
else
{
lean_inc(v_a_927_);
lean_dec(v___x_906_);
v___x_929_ = lean_box(0);
v_isShared_930_ = v_isSharedCheck_934_;
goto v_resetjp_928_;
}
v_resetjp_928_:
{
lean_object* v___x_932_; 
if (v_isShared_930_ == 0)
{
v___x_932_ = v___x_929_;
goto v_reusejp_931_;
}
else
{
lean_object* v_reuseFailAlloc_933_; 
v_reuseFailAlloc_933_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_933_, 0, v_a_927_);
v___x_932_ = v_reuseFailAlloc_933_;
goto v_reusejp_931_;
}
v_reusejp_931_:
{
return v___x_932_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_Frontend_addLocalImplications_spec__0_spec__0_spec__1___boxed(lean_object* v_init_937_, lean_object* v_as_938_, lean_object* v_sz_939_, lean_object* v_i_940_, lean_object* v_b_941_, lean_object* v___y_942_, lean_object* v___y_943_, lean_object* v___y_944_, lean_object* v___y_945_, lean_object* v___y_946_, lean_object* v___y_947_, lean_object* v___y_948_, lean_object* v___y_949_){
_start:
{
size_t v_sz_boxed_950_; size_t v_i_boxed_951_; lean_object* v_res_952_; 
v_sz_boxed_950_ = lean_unbox_usize(v_sz_939_);
lean_dec(v_sz_939_);
v_i_boxed_951_ = lean_unbox_usize(v_i_940_);
lean_dec(v_i_940_);
v_res_952_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_Frontend_addLocalImplications_spec__0_spec__0_spec__1(v_init_937_, v_as_938_, v_sz_boxed_950_, v_i_boxed_951_, v_b_941_, v___y_942_, v___y_943_, v___y_944_, v___y_945_, v___y_946_, v___y_947_, v___y_948_);
lean_dec(v___y_948_);
lean_dec_ref(v___y_947_);
lean_dec(v___y_946_);
lean_dec_ref(v___y_945_);
lean_dec(v___y_944_);
lean_dec_ref(v___y_943_);
lean_dec_ref(v___y_942_);
lean_dec_ref(v_as_938_);
lean_dec_ref(v_init_937_);
return v_res_952_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_Frontend_addLocalImplications_spec__0_spec__0___boxed(lean_object* v_init_953_, lean_object* v_n_954_, lean_object* v_b_955_, lean_object* v___y_956_, lean_object* v___y_957_, lean_object* v___y_958_, lean_object* v___y_959_, lean_object* v___y_960_, lean_object* v___y_961_, lean_object* v___y_962_, lean_object* v___y_963_){
_start:
{
lean_object* v_res_964_; 
v_res_964_ = lp_aesop_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_Frontend_addLocalImplications_spec__0_spec__0(v_init_953_, v_n_954_, v_b_955_, v___y_956_, v___y_957_, v___y_958_, v___y_959_, v___y_960_, v___y_961_, v___y_962_);
lean_dec(v___y_962_);
lean_dec_ref(v___y_961_);
lean_dec(v___y_960_);
lean_dec_ref(v___y_959_);
lean_dec(v___y_958_);
lean_dec_ref(v___y_957_);
lean_dec_ref(v___y_956_);
lean_dec_ref(v_n_954_);
lean_dec_ref(v_init_953_);
return v_res_964_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Aesop_Frontend_addLocalImplications_spec__0_spec__1_spec__4(lean_object* v_as_965_, size_t v_sz_966_, size_t v_i_967_, lean_object* v_b_968_, lean_object* v___y_969_, lean_object* v___y_970_, lean_object* v___y_971_, lean_object* v___y_972_, lean_object* v___y_973_, lean_object* v___y_974_, lean_object* v___y_975_){
_start:
{
uint8_t v___x_977_; 
v___x_977_ = lean_usize_dec_lt(v_i_967_, v_sz_966_);
if (v___x_977_ == 0)
{
lean_object* v___x_978_; 
v___x_978_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_978_, 0, v_b_968_);
return v___x_978_;
}
else
{
lean_object* v_snd_979_; lean_object* v___x_981_; uint8_t v_isShared_982_; uint8_t v_isSharedCheck_1008_; 
v_snd_979_ = lean_ctor_get(v_b_968_, 1);
v_isSharedCheck_1008_ = !lean_is_exclusive(v_b_968_);
if (v_isSharedCheck_1008_ == 0)
{
lean_object* v_unused_1009_; 
v_unused_1009_ = lean_ctor_get(v_b_968_, 0);
lean_dec(v_unused_1009_);
v___x_981_ = v_b_968_;
v_isShared_982_ = v_isSharedCheck_1008_;
goto v_resetjp_980_;
}
else
{
lean_inc(v_snd_979_);
lean_dec(v_b_968_);
v___x_981_ = lean_box(0);
v_isShared_982_ = v_isSharedCheck_1008_;
goto v_resetjp_980_;
}
v_resetjp_980_:
{
lean_object* v___x_983_; lean_object* v_a_985_; lean_object* v_a_992_; 
v___x_983_ = lean_box(0);
v_a_992_ = lean_array_uget_borrowed(v_as_965_, v_i_967_);
if (lean_obj_tag(v_a_992_) == 0)
{
v_a_985_ = v_snd_979_;
goto v___jp_984_;
}
else
{
lean_object* v_val_993_; uint8_t v___x_994_; 
v_val_993_ = lean_ctor_get(v_a_992_, 0);
v___x_994_ = l_Lean_LocalDecl_isImplementationDetail(v_val_993_);
if (v___x_994_ == 0)
{
lean_object* v___x_995_; lean_object* v___x_996_; 
v___x_995_ = l_Lean_LocalDecl_fvarId(v_val_993_);
v___x_996_ = lp_aesop_Aesop_Frontend_mkHypImplicationRule_x3f(v___x_995_, v___y_969_, v___y_970_, v___y_971_, v___y_972_, v___y_973_, v___y_974_, v___y_975_);
if (lean_obj_tag(v___x_996_) == 0)
{
lean_object* v_a_997_; 
v_a_997_ = lean_ctor_get(v___x_996_, 0);
lean_inc(v_a_997_);
lean_dec_ref_known(v___x_996_, 1);
if (lean_obj_tag(v_a_997_) == 1)
{
lean_object* v_val_998_; lean_object* v___x_999_; 
v_val_998_ = lean_ctor_get(v_a_997_, 0);
lean_inc(v_val_998_);
lean_dec_ref_known(v_a_997_, 1);
v___x_999_ = lp_aesop_Aesop_LocalRuleSet_add(v_snd_979_, v_val_998_);
v_a_985_ = v___x_999_;
goto v___jp_984_;
}
else
{
lean_dec(v_a_997_);
v_a_985_ = v_snd_979_;
goto v___jp_984_;
}
}
else
{
lean_object* v_a_1000_; lean_object* v___x_1002_; uint8_t v_isShared_1003_; uint8_t v_isSharedCheck_1007_; 
lean_del_object(v___x_981_);
lean_dec(v_snd_979_);
v_a_1000_ = lean_ctor_get(v___x_996_, 0);
v_isSharedCheck_1007_ = !lean_is_exclusive(v___x_996_);
if (v_isSharedCheck_1007_ == 0)
{
v___x_1002_ = v___x_996_;
v_isShared_1003_ = v_isSharedCheck_1007_;
goto v_resetjp_1001_;
}
else
{
lean_inc(v_a_1000_);
lean_dec(v___x_996_);
v___x_1002_ = lean_box(0);
v_isShared_1003_ = v_isSharedCheck_1007_;
goto v_resetjp_1001_;
}
v_resetjp_1001_:
{
lean_object* v___x_1005_; 
if (v_isShared_1003_ == 0)
{
v___x_1005_ = v___x_1002_;
goto v_reusejp_1004_;
}
else
{
lean_object* v_reuseFailAlloc_1006_; 
v_reuseFailAlloc_1006_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1006_, 0, v_a_1000_);
v___x_1005_ = v_reuseFailAlloc_1006_;
goto v_reusejp_1004_;
}
v_reusejp_1004_:
{
return v___x_1005_;
}
}
}
}
else
{
v_a_985_ = v_snd_979_;
goto v___jp_984_;
}
}
v___jp_984_:
{
lean_object* v___x_987_; 
if (v_isShared_982_ == 0)
{
lean_ctor_set(v___x_981_, 1, v_a_985_);
lean_ctor_set(v___x_981_, 0, v___x_983_);
v___x_987_ = v___x_981_;
goto v_reusejp_986_;
}
else
{
lean_object* v_reuseFailAlloc_991_; 
v_reuseFailAlloc_991_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_991_, 0, v___x_983_);
lean_ctor_set(v_reuseFailAlloc_991_, 1, v_a_985_);
v___x_987_ = v_reuseFailAlloc_991_;
goto v_reusejp_986_;
}
v_reusejp_986_:
{
size_t v___x_988_; size_t v___x_989_; 
v___x_988_ = ((size_t)1ULL);
v___x_989_ = lean_usize_add(v_i_967_, v___x_988_);
v_i_967_ = v___x_989_;
v_b_968_ = v___x_987_;
goto _start;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Aesop_Frontend_addLocalImplications_spec__0_spec__1_spec__4___boxed(lean_object* v_as_1010_, lean_object* v_sz_1011_, lean_object* v_i_1012_, lean_object* v_b_1013_, lean_object* v___y_1014_, lean_object* v___y_1015_, lean_object* v___y_1016_, lean_object* v___y_1017_, lean_object* v___y_1018_, lean_object* v___y_1019_, lean_object* v___y_1020_, lean_object* v___y_1021_){
_start:
{
size_t v_sz_boxed_1022_; size_t v_i_boxed_1023_; lean_object* v_res_1024_; 
v_sz_boxed_1022_ = lean_unbox_usize(v_sz_1011_);
lean_dec(v_sz_1011_);
v_i_boxed_1023_ = lean_unbox_usize(v_i_1012_);
lean_dec(v_i_1012_);
v_res_1024_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Aesop_Frontend_addLocalImplications_spec__0_spec__1_spec__4(v_as_1010_, v_sz_boxed_1022_, v_i_boxed_1023_, v_b_1013_, v___y_1014_, v___y_1015_, v___y_1016_, v___y_1017_, v___y_1018_, v___y_1019_, v___y_1020_);
lean_dec(v___y_1020_);
lean_dec_ref(v___y_1019_);
lean_dec(v___y_1018_);
lean_dec_ref(v___y_1017_);
lean_dec(v___y_1016_);
lean_dec_ref(v___y_1015_);
lean_dec_ref(v___y_1014_);
lean_dec_ref(v_as_1010_);
return v_res_1024_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Aesop_Frontend_addLocalImplications_spec__0_spec__1(lean_object* v_as_1025_, size_t v_sz_1026_, size_t v_i_1027_, lean_object* v_b_1028_, lean_object* v___y_1029_, lean_object* v___y_1030_, lean_object* v___y_1031_, lean_object* v___y_1032_, lean_object* v___y_1033_, lean_object* v___y_1034_, lean_object* v___y_1035_){
_start:
{
uint8_t v___x_1037_; 
v___x_1037_ = lean_usize_dec_lt(v_i_1027_, v_sz_1026_);
if (v___x_1037_ == 0)
{
lean_object* v___x_1038_; 
v___x_1038_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1038_, 0, v_b_1028_);
return v___x_1038_;
}
else
{
lean_object* v_snd_1039_; lean_object* v___x_1041_; uint8_t v_isShared_1042_; uint8_t v_isSharedCheck_1068_; 
v_snd_1039_ = lean_ctor_get(v_b_1028_, 1);
v_isSharedCheck_1068_ = !lean_is_exclusive(v_b_1028_);
if (v_isSharedCheck_1068_ == 0)
{
lean_object* v_unused_1069_; 
v_unused_1069_ = lean_ctor_get(v_b_1028_, 0);
lean_dec(v_unused_1069_);
v___x_1041_ = v_b_1028_;
v_isShared_1042_ = v_isSharedCheck_1068_;
goto v_resetjp_1040_;
}
else
{
lean_inc(v_snd_1039_);
lean_dec(v_b_1028_);
v___x_1041_ = lean_box(0);
v_isShared_1042_ = v_isSharedCheck_1068_;
goto v_resetjp_1040_;
}
v_resetjp_1040_:
{
lean_object* v___x_1043_; lean_object* v_a_1045_; lean_object* v_a_1052_; 
v___x_1043_ = lean_box(0);
v_a_1052_ = lean_array_uget_borrowed(v_as_1025_, v_i_1027_);
if (lean_obj_tag(v_a_1052_) == 0)
{
v_a_1045_ = v_snd_1039_;
goto v___jp_1044_;
}
else
{
lean_object* v_val_1053_; uint8_t v___x_1054_; 
v_val_1053_ = lean_ctor_get(v_a_1052_, 0);
v___x_1054_ = l_Lean_LocalDecl_isImplementationDetail(v_val_1053_);
if (v___x_1054_ == 0)
{
lean_object* v___x_1055_; lean_object* v___x_1056_; 
v___x_1055_ = l_Lean_LocalDecl_fvarId(v_val_1053_);
v___x_1056_ = lp_aesop_Aesop_Frontend_mkHypImplicationRule_x3f(v___x_1055_, v___y_1029_, v___y_1030_, v___y_1031_, v___y_1032_, v___y_1033_, v___y_1034_, v___y_1035_);
if (lean_obj_tag(v___x_1056_) == 0)
{
lean_object* v_a_1057_; 
v_a_1057_ = lean_ctor_get(v___x_1056_, 0);
lean_inc(v_a_1057_);
lean_dec_ref_known(v___x_1056_, 1);
if (lean_obj_tag(v_a_1057_) == 1)
{
lean_object* v_val_1058_; lean_object* v___x_1059_; 
v_val_1058_ = lean_ctor_get(v_a_1057_, 0);
lean_inc(v_val_1058_);
lean_dec_ref_known(v_a_1057_, 1);
v___x_1059_ = lp_aesop_Aesop_LocalRuleSet_add(v_snd_1039_, v_val_1058_);
v_a_1045_ = v___x_1059_;
goto v___jp_1044_;
}
else
{
lean_dec(v_a_1057_);
v_a_1045_ = v_snd_1039_;
goto v___jp_1044_;
}
}
else
{
lean_object* v_a_1060_; lean_object* v___x_1062_; uint8_t v_isShared_1063_; uint8_t v_isSharedCheck_1067_; 
lean_del_object(v___x_1041_);
lean_dec(v_snd_1039_);
v_a_1060_ = lean_ctor_get(v___x_1056_, 0);
v_isSharedCheck_1067_ = !lean_is_exclusive(v___x_1056_);
if (v_isSharedCheck_1067_ == 0)
{
v___x_1062_ = v___x_1056_;
v_isShared_1063_ = v_isSharedCheck_1067_;
goto v_resetjp_1061_;
}
else
{
lean_inc(v_a_1060_);
lean_dec(v___x_1056_);
v___x_1062_ = lean_box(0);
v_isShared_1063_ = v_isSharedCheck_1067_;
goto v_resetjp_1061_;
}
v_resetjp_1061_:
{
lean_object* v___x_1065_; 
if (v_isShared_1063_ == 0)
{
v___x_1065_ = v___x_1062_;
goto v_reusejp_1064_;
}
else
{
lean_object* v_reuseFailAlloc_1066_; 
v_reuseFailAlloc_1066_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1066_, 0, v_a_1060_);
v___x_1065_ = v_reuseFailAlloc_1066_;
goto v_reusejp_1064_;
}
v_reusejp_1064_:
{
return v___x_1065_;
}
}
}
}
else
{
v_a_1045_ = v_snd_1039_;
goto v___jp_1044_;
}
}
v___jp_1044_:
{
lean_object* v___x_1047_; 
if (v_isShared_1042_ == 0)
{
lean_ctor_set(v___x_1041_, 1, v_a_1045_);
lean_ctor_set(v___x_1041_, 0, v___x_1043_);
v___x_1047_ = v___x_1041_;
goto v_reusejp_1046_;
}
else
{
lean_object* v_reuseFailAlloc_1051_; 
v_reuseFailAlloc_1051_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1051_, 0, v___x_1043_);
lean_ctor_set(v_reuseFailAlloc_1051_, 1, v_a_1045_);
v___x_1047_ = v_reuseFailAlloc_1051_;
goto v_reusejp_1046_;
}
v_reusejp_1046_:
{
size_t v___x_1048_; size_t v___x_1049_; lean_object* v___x_1050_; 
v___x_1048_ = ((size_t)1ULL);
v___x_1049_ = lean_usize_add(v_i_1027_, v___x_1048_);
v___x_1050_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Aesop_Frontend_addLocalImplications_spec__0_spec__1_spec__4(v_as_1025_, v_sz_1026_, v___x_1049_, v___x_1047_, v___y_1029_, v___y_1030_, v___y_1031_, v___y_1032_, v___y_1033_, v___y_1034_, v___y_1035_);
return v___x_1050_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Aesop_Frontend_addLocalImplications_spec__0_spec__1___boxed(lean_object* v_as_1070_, lean_object* v_sz_1071_, lean_object* v_i_1072_, lean_object* v_b_1073_, lean_object* v___y_1074_, lean_object* v___y_1075_, lean_object* v___y_1076_, lean_object* v___y_1077_, lean_object* v___y_1078_, lean_object* v___y_1079_, lean_object* v___y_1080_, lean_object* v___y_1081_){
_start:
{
size_t v_sz_boxed_1082_; size_t v_i_boxed_1083_; lean_object* v_res_1084_; 
v_sz_boxed_1082_ = lean_unbox_usize(v_sz_1071_);
lean_dec(v_sz_1071_);
v_i_boxed_1083_ = lean_unbox_usize(v_i_1072_);
lean_dec(v_i_1072_);
v_res_1084_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Aesop_Frontend_addLocalImplications_spec__0_spec__1(v_as_1070_, v_sz_boxed_1082_, v_i_boxed_1083_, v_b_1073_, v___y_1074_, v___y_1075_, v___y_1076_, v___y_1077_, v___y_1078_, v___y_1079_, v___y_1080_);
lean_dec(v___y_1080_);
lean_dec_ref(v___y_1079_);
lean_dec(v___y_1078_);
lean_dec_ref(v___y_1077_);
lean_dec(v___y_1076_);
lean_dec_ref(v___y_1075_);
lean_dec_ref(v___y_1074_);
lean_dec_ref(v_as_1070_);
return v_res_1084_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentArray_forIn___at___00Aesop_Frontend_addLocalImplications_spec__0(lean_object* v_t_1085_, lean_object* v_init_1086_, lean_object* v___y_1087_, lean_object* v___y_1088_, lean_object* v___y_1089_, lean_object* v___y_1090_, lean_object* v___y_1091_, lean_object* v___y_1092_, lean_object* v___y_1093_){
_start:
{
lean_object* v_root_1095_; lean_object* v_tail_1096_; lean_object* v___x_1097_; 
v_root_1095_ = lean_ctor_get(v_t_1085_, 0);
v_tail_1096_ = lean_ctor_get(v_t_1085_, 1);
lean_inc_ref(v_init_1086_);
v___x_1097_ = lp_aesop_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_Frontend_addLocalImplications_spec__0_spec__0(v_init_1086_, v_root_1095_, v_init_1086_, v___y_1087_, v___y_1088_, v___y_1089_, v___y_1090_, v___y_1091_, v___y_1092_, v___y_1093_);
lean_dec_ref(v_init_1086_);
if (lean_obj_tag(v___x_1097_) == 0)
{
lean_object* v_a_1098_; lean_object* v___x_1100_; uint8_t v_isShared_1101_; uint8_t v_isSharedCheck_1134_; 
v_a_1098_ = lean_ctor_get(v___x_1097_, 0);
v_isSharedCheck_1134_ = !lean_is_exclusive(v___x_1097_);
if (v_isSharedCheck_1134_ == 0)
{
v___x_1100_ = v___x_1097_;
v_isShared_1101_ = v_isSharedCheck_1134_;
goto v_resetjp_1099_;
}
else
{
lean_inc(v_a_1098_);
lean_dec(v___x_1097_);
v___x_1100_ = lean_box(0);
v_isShared_1101_ = v_isSharedCheck_1134_;
goto v_resetjp_1099_;
}
v_resetjp_1099_:
{
if (lean_obj_tag(v_a_1098_) == 0)
{
lean_object* v_a_1102_; lean_object* v___x_1104_; 
v_a_1102_ = lean_ctor_get(v_a_1098_, 0);
lean_inc(v_a_1102_);
lean_dec_ref_known(v_a_1098_, 1);
if (v_isShared_1101_ == 0)
{
lean_ctor_set(v___x_1100_, 0, v_a_1102_);
v___x_1104_ = v___x_1100_;
goto v_reusejp_1103_;
}
else
{
lean_object* v_reuseFailAlloc_1105_; 
v_reuseFailAlloc_1105_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1105_, 0, v_a_1102_);
v___x_1104_ = v_reuseFailAlloc_1105_;
goto v_reusejp_1103_;
}
v_reusejp_1103_:
{
return v___x_1104_;
}
}
else
{
lean_object* v_a_1106_; lean_object* v___x_1107_; lean_object* v___x_1108_; size_t v_sz_1109_; size_t v___x_1110_; lean_object* v___x_1111_; 
lean_del_object(v___x_1100_);
v_a_1106_ = lean_ctor_get(v_a_1098_, 0);
lean_inc(v_a_1106_);
lean_dec_ref_known(v_a_1098_, 1);
v___x_1107_ = lean_box(0);
v___x_1108_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1108_, 0, v___x_1107_);
lean_ctor_set(v___x_1108_, 1, v_a_1106_);
v_sz_1109_ = lean_array_size(v_tail_1096_);
v___x_1110_ = ((size_t)0ULL);
v___x_1111_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Aesop_Frontend_addLocalImplications_spec__0_spec__1(v_tail_1096_, v_sz_1109_, v___x_1110_, v___x_1108_, v___y_1087_, v___y_1088_, v___y_1089_, v___y_1090_, v___y_1091_, v___y_1092_, v___y_1093_);
if (lean_obj_tag(v___x_1111_) == 0)
{
lean_object* v_a_1112_; lean_object* v___x_1114_; uint8_t v_isShared_1115_; uint8_t v_isSharedCheck_1125_; 
v_a_1112_ = lean_ctor_get(v___x_1111_, 0);
v_isSharedCheck_1125_ = !lean_is_exclusive(v___x_1111_);
if (v_isSharedCheck_1125_ == 0)
{
v___x_1114_ = v___x_1111_;
v_isShared_1115_ = v_isSharedCheck_1125_;
goto v_resetjp_1113_;
}
else
{
lean_inc(v_a_1112_);
lean_dec(v___x_1111_);
v___x_1114_ = lean_box(0);
v_isShared_1115_ = v_isSharedCheck_1125_;
goto v_resetjp_1113_;
}
v_resetjp_1113_:
{
lean_object* v_fst_1116_; 
v_fst_1116_ = lean_ctor_get(v_a_1112_, 0);
if (lean_obj_tag(v_fst_1116_) == 0)
{
lean_object* v_snd_1117_; lean_object* v___x_1119_; 
v_snd_1117_ = lean_ctor_get(v_a_1112_, 1);
lean_inc(v_snd_1117_);
lean_dec(v_a_1112_);
if (v_isShared_1115_ == 0)
{
lean_ctor_set(v___x_1114_, 0, v_snd_1117_);
v___x_1119_ = v___x_1114_;
goto v_reusejp_1118_;
}
else
{
lean_object* v_reuseFailAlloc_1120_; 
v_reuseFailAlloc_1120_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1120_, 0, v_snd_1117_);
v___x_1119_ = v_reuseFailAlloc_1120_;
goto v_reusejp_1118_;
}
v_reusejp_1118_:
{
return v___x_1119_;
}
}
else
{
lean_object* v_val_1121_; lean_object* v___x_1123_; 
lean_inc_ref(v_fst_1116_);
lean_dec(v_a_1112_);
v_val_1121_ = lean_ctor_get(v_fst_1116_, 0);
lean_inc(v_val_1121_);
lean_dec_ref_known(v_fst_1116_, 1);
if (v_isShared_1115_ == 0)
{
lean_ctor_set(v___x_1114_, 0, v_val_1121_);
v___x_1123_ = v___x_1114_;
goto v_reusejp_1122_;
}
else
{
lean_object* v_reuseFailAlloc_1124_; 
v_reuseFailAlloc_1124_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1124_, 0, v_val_1121_);
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
else
{
lean_object* v_a_1126_; lean_object* v___x_1128_; uint8_t v_isShared_1129_; uint8_t v_isSharedCheck_1133_; 
v_a_1126_ = lean_ctor_get(v___x_1111_, 0);
v_isSharedCheck_1133_ = !lean_is_exclusive(v___x_1111_);
if (v_isSharedCheck_1133_ == 0)
{
v___x_1128_ = v___x_1111_;
v_isShared_1129_ = v_isSharedCheck_1133_;
goto v_resetjp_1127_;
}
else
{
lean_inc(v_a_1126_);
lean_dec(v___x_1111_);
v___x_1128_ = lean_box(0);
v_isShared_1129_ = v_isSharedCheck_1133_;
goto v_resetjp_1127_;
}
v_resetjp_1127_:
{
lean_object* v___x_1131_; 
if (v_isShared_1129_ == 0)
{
v___x_1131_ = v___x_1128_;
goto v_reusejp_1130_;
}
else
{
lean_object* v_reuseFailAlloc_1132_; 
v_reuseFailAlloc_1132_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1132_, 0, v_a_1126_);
v___x_1131_ = v_reuseFailAlloc_1132_;
goto v_reusejp_1130_;
}
v_reusejp_1130_:
{
return v___x_1131_;
}
}
}
}
}
}
else
{
lean_object* v_a_1135_; lean_object* v___x_1137_; uint8_t v_isShared_1138_; uint8_t v_isSharedCheck_1142_; 
v_a_1135_ = lean_ctor_get(v___x_1097_, 0);
v_isSharedCheck_1142_ = !lean_is_exclusive(v___x_1097_);
if (v_isSharedCheck_1142_ == 0)
{
v___x_1137_ = v___x_1097_;
v_isShared_1138_ = v_isSharedCheck_1142_;
goto v_resetjp_1136_;
}
else
{
lean_inc(v_a_1135_);
lean_dec(v___x_1097_);
v___x_1137_ = lean_box(0);
v_isShared_1138_ = v_isSharedCheck_1142_;
goto v_resetjp_1136_;
}
v_resetjp_1136_:
{
lean_object* v___x_1140_; 
if (v_isShared_1138_ == 0)
{
v___x_1140_ = v___x_1137_;
goto v_reusejp_1139_;
}
else
{
lean_object* v_reuseFailAlloc_1141_; 
v_reuseFailAlloc_1141_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1141_, 0, v_a_1135_);
v___x_1140_ = v_reuseFailAlloc_1141_;
goto v_reusejp_1139_;
}
v_reusejp_1139_:
{
return v___x_1140_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentArray_forIn___at___00Aesop_Frontend_addLocalImplications_spec__0___boxed(lean_object* v_t_1143_, lean_object* v_init_1144_, lean_object* v___y_1145_, lean_object* v___y_1146_, lean_object* v___y_1147_, lean_object* v___y_1148_, lean_object* v___y_1149_, lean_object* v___y_1150_, lean_object* v___y_1151_, lean_object* v___y_1152_){
_start:
{
lean_object* v_res_1153_; 
v_res_1153_ = lp_aesop_Lean_PersistentArray_forIn___at___00Aesop_Frontend_addLocalImplications_spec__0(v_t_1143_, v_init_1144_, v___y_1145_, v___y_1146_, v___y_1147_, v___y_1148_, v___y_1149_, v___y_1150_, v___y_1151_);
lean_dec(v___y_1151_);
lean_dec_ref(v___y_1150_);
lean_dec(v___y_1149_);
lean_dec_ref(v___y_1148_);
lean_dec(v___y_1147_);
lean_dec_ref(v___y_1146_);
lean_dec_ref(v___y_1145_);
lean_dec_ref(v_t_1143_);
return v_res_1153_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_addLocalImplications(lean_object* v_rs_1154_, lean_object* v_a_1155_, lean_object* v_a_1156_, lean_object* v_a_1157_, lean_object* v_a_1158_, lean_object* v_a_1159_, lean_object* v_a_1160_, lean_object* v_a_1161_){
_start:
{
lean_object* v_goal_1163_; lean_object* v___x_1164_; 
v_goal_1163_ = lean_ctor_get(v_a_1155_, 0);
lean_inc(v_goal_1163_);
v___x_1164_ = l_Lean_MVarId_getDecl(v_goal_1163_, v_a_1158_, v_a_1159_, v_a_1160_, v_a_1161_);
if (lean_obj_tag(v___x_1164_) == 0)
{
lean_object* v_a_1165_; lean_object* v_lctx_1166_; lean_object* v_decls_1167_; lean_object* v___x_1168_; 
v_a_1165_ = lean_ctor_get(v___x_1164_, 0);
lean_inc(v_a_1165_);
lean_dec_ref_known(v___x_1164_, 1);
v_lctx_1166_ = lean_ctor_get(v_a_1165_, 1);
lean_inc_ref(v_lctx_1166_);
lean_dec(v_a_1165_);
v_decls_1167_ = lean_ctor_get(v_lctx_1166_, 1);
lean_inc_ref(v_decls_1167_);
lean_dec_ref(v_lctx_1166_);
v___x_1168_ = lp_aesop_Lean_PersistentArray_forIn___at___00Aesop_Frontend_addLocalImplications_spec__0(v_decls_1167_, v_rs_1154_, v_a_1155_, v_a_1156_, v_a_1157_, v_a_1158_, v_a_1159_, v_a_1160_, v_a_1161_);
lean_dec_ref(v_decls_1167_);
return v___x_1168_;
}
else
{
lean_object* v_a_1169_; lean_object* v___x_1171_; uint8_t v_isShared_1172_; uint8_t v_isSharedCheck_1176_; 
lean_dec_ref(v_rs_1154_);
v_a_1169_ = lean_ctor_get(v___x_1164_, 0);
v_isSharedCheck_1176_ = !lean_is_exclusive(v___x_1164_);
if (v_isSharedCheck_1176_ == 0)
{
v___x_1171_ = v___x_1164_;
v_isShared_1172_ = v_isSharedCheck_1176_;
goto v_resetjp_1170_;
}
else
{
lean_inc(v_a_1169_);
lean_dec(v___x_1164_);
v___x_1171_ = lean_box(0);
v_isShared_1172_ = v_isSharedCheck_1176_;
goto v_resetjp_1170_;
}
v_resetjp_1170_:
{
lean_object* v___x_1174_; 
if (v_isShared_1172_ == 0)
{
v___x_1174_ = v___x_1171_;
goto v_reusejp_1173_;
}
else
{
lean_object* v_reuseFailAlloc_1175_; 
v_reuseFailAlloc_1175_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1175_, 0, v_a_1169_);
v___x_1174_ = v_reuseFailAlloc_1175_;
goto v_reusejp_1173_;
}
v_reusejp_1173_:
{
return v___x_1174_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_addLocalImplications___boxed(lean_object* v_rs_1177_, lean_object* v_a_1178_, lean_object* v_a_1179_, lean_object* v_a_1180_, lean_object* v_a_1181_, lean_object* v_a_1182_, lean_object* v_a_1183_, lean_object* v_a_1184_, lean_object* v_a_1185_){
_start:
{
lean_object* v_res_1186_; 
v_res_1186_ = lp_aesop_Aesop_Frontend_addLocalImplications(v_rs_1177_, v_a_1178_, v_a_1179_, v_a_1180_, v_a_1181_, v_a_1182_, v_a_1183_, v_a_1184_);
lean_dec(v_a_1184_);
lean_dec_ref(v_a_1183_);
lean_dec(v_a_1182_);
lean_dec_ref(v_a_1181_);
lean_dec(v_a_1180_);
lean_dec_ref(v_a_1179_);
lean_dec_ref(v_a_1178_);
return v_res_1186_;
}
}
static lean_object* _init_lp_aesop_Lean_Elab_throwUnsupportedSyntax___at___00Aesop_Frontend_elabAdditionalForwardRules_spec__0___redArg___closed__0(void){
_start:
{
lean_object* v___x_1187_; lean_object* v___x_1188_; lean_object* v___x_1189_; 
v___x_1187_ = lean_box(0);
v___x_1188_ = l_Lean_Elab_unsupportedSyntaxExceptionId;
v___x_1189_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1189_, 0, v___x_1188_);
lean_ctor_set(v___x_1189_, 1, v___x_1187_);
return v___x_1189_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Elab_throwUnsupportedSyntax___at___00Aesop_Frontend_elabAdditionalForwardRules_spec__0___redArg(){
_start:
{
lean_object* v___x_1191_; lean_object* v___x_1192_; 
v___x_1191_ = lean_obj_once(&lp_aesop_Lean_Elab_throwUnsupportedSyntax___at___00Aesop_Frontend_elabAdditionalForwardRules_spec__0___redArg___closed__0, &lp_aesop_Lean_Elab_throwUnsupportedSyntax___at___00Aesop_Frontend_elabAdditionalForwardRules_spec__0___redArg___closed__0_once, _init_lp_aesop_Lean_Elab_throwUnsupportedSyntax___at___00Aesop_Frontend_elabAdditionalForwardRules_spec__0___redArg___closed__0);
v___x_1192_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1192_, 0, v___x_1191_);
return v___x_1192_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Elab_throwUnsupportedSyntax___at___00Aesop_Frontend_elabAdditionalForwardRules_spec__0___redArg___boxed(lean_object* v___y_1193_){
_start:
{
lean_object* v_res_1194_; 
v_res_1194_ = lp_aesop_Lean_Elab_throwUnsupportedSyntax___at___00Aesop_Frontend_elabAdditionalForwardRules_spec__0___redArg();
return v_res_1194_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Elab_throwUnsupportedSyntax___at___00Aesop_Frontend_elabAdditionalForwardRules_spec__0(lean_object* v_00_u03b1_1195_, lean_object* v___y_1196_, lean_object* v___y_1197_, lean_object* v___y_1198_, lean_object* v___y_1199_, lean_object* v___y_1200_, lean_object* v___y_1201_, lean_object* v___y_1202_){
_start:
{
lean_object* v___x_1204_; 
v___x_1204_ = lp_aesop_Lean_Elab_throwUnsupportedSyntax___at___00Aesop_Frontend_elabAdditionalForwardRules_spec__0___redArg();
return v___x_1204_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Elab_throwUnsupportedSyntax___at___00Aesop_Frontend_elabAdditionalForwardRules_spec__0___boxed(lean_object* v_00_u03b1_1205_, lean_object* v___y_1206_, lean_object* v___y_1207_, lean_object* v___y_1208_, lean_object* v___y_1209_, lean_object* v___y_1210_, lean_object* v___y_1211_, lean_object* v___y_1212_, lean_object* v___y_1213_){
_start:
{
lean_object* v_res_1214_; 
v_res_1214_ = lp_aesop_Lean_Elab_throwUnsupportedSyntax___at___00Aesop_Frontend_elabAdditionalForwardRules_spec__0(v_00_u03b1_1205_, v___y_1206_, v___y_1207_, v___y_1208_, v___y_1209_, v___y_1210_, v___y_1211_, v___y_1212_);
lean_dec(v___y_1212_);
lean_dec_ref(v___y_1211_);
lean_dec(v___y_1210_);
lean_dec_ref(v___y_1209_);
lean_dec(v___y_1208_);
lean_dec_ref(v___y_1207_);
lean_dec_ref(v___y_1206_);
return v_res_1214_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Frontend_elabAdditionalForwardRules_spec__1(lean_object* v_as_1215_, size_t v_sz_1216_, size_t v_i_1217_, lean_object* v_b_1218_, lean_object* v___y_1219_, lean_object* v___y_1220_, lean_object* v___y_1221_, lean_object* v___y_1222_, lean_object* v___y_1223_, lean_object* v___y_1224_, lean_object* v___y_1225_){
_start:
{
lean_object* v_a_1228_; uint8_t v___x_1232_; 
v___x_1232_ = lean_usize_dec_lt(v_i_1217_, v_sz_1216_);
if (v___x_1232_ == 0)
{
lean_object* v___x_1233_; 
v___x_1233_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1233_, 0, v_b_1218_);
return v___x_1233_;
}
else
{
lean_object* v_fst_1234_; lean_object* v_snd_1235_; lean_object* v___x_1237_; uint8_t v_isShared_1238_; uint8_t v_isSharedCheck_1276_; 
v_fst_1234_ = lean_ctor_get(v_b_1218_, 0);
v_snd_1235_ = lean_ctor_get(v_b_1218_, 1);
v_isSharedCheck_1276_ = !lean_is_exclusive(v_b_1218_);
if (v_isSharedCheck_1276_ == 0)
{
v___x_1237_ = v_b_1218_;
v_isShared_1238_ = v_isSharedCheck_1276_;
goto v_resetjp_1236_;
}
else
{
lean_inc(v_snd_1235_);
lean_inc(v_fst_1234_);
lean_dec(v_b_1218_);
v___x_1237_ = lean_box(0);
v_isShared_1238_ = v_isSharedCheck_1276_;
goto v_resetjp_1236_;
}
v_resetjp_1236_:
{
lean_object* v_a_1239_; lean_object* v___x_1240_; uint8_t v___x_1241_; 
v_a_1239_ = lean_array_uget_borrowed(v_as_1215_, v_i_1217_);
v___x_1240_ = ((lean_object*)(lp_aesop_Aesop_Frontend_Parser_additionalRule___closed__1));
lean_inc(v_a_1239_);
v___x_1241_ = l_Lean_Syntax_isOfKind(v_a_1239_, v___x_1240_);
if (v___x_1241_ == 0)
{
lean_object* v___x_1242_; 
v___x_1242_ = lp_aesop_Lean_Elab_throwUnsupportedSyntax___at___00Aesop_Frontend_elabAdditionalForwardRules_spec__0___redArg();
if (lean_obj_tag(v___x_1242_) == 0)
{
lean_object* v___x_1244_; 
lean_dec_ref_known(v___x_1242_, 1);
if (v_isShared_1238_ == 0)
{
v___x_1244_ = v___x_1237_;
goto v_reusejp_1243_;
}
else
{
lean_object* v_reuseFailAlloc_1245_; 
v_reuseFailAlloc_1245_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1245_, 0, v_fst_1234_);
lean_ctor_set(v_reuseFailAlloc_1245_, 1, v_snd_1235_);
v___x_1244_ = v_reuseFailAlloc_1245_;
goto v_reusejp_1243_;
}
v_reusejp_1243_:
{
v_a_1228_ = v___x_1244_;
goto v___jp_1227_;
}
}
else
{
lean_object* v_a_1246_; lean_object* v___x_1248_; uint8_t v_isShared_1249_; uint8_t v_isSharedCheck_1253_; 
lean_del_object(v___x_1237_);
lean_dec(v_snd_1235_);
lean_dec(v_fst_1234_);
v_a_1246_ = lean_ctor_get(v___x_1242_, 0);
v_isSharedCheck_1253_ = !lean_is_exclusive(v___x_1242_);
if (v_isSharedCheck_1253_ == 0)
{
v___x_1248_ = v___x_1242_;
v_isShared_1249_ = v_isSharedCheck_1253_;
goto v_resetjp_1247_;
}
else
{
lean_inc(v_a_1246_);
lean_dec(v___x_1242_);
v___x_1248_ = lean_box(0);
v_isShared_1249_ = v_isSharedCheck_1253_;
goto v_resetjp_1247_;
}
v_resetjp_1247_:
{
lean_object* v___x_1251_; 
if (v_isShared_1249_ == 0)
{
v___x_1251_ = v___x_1248_;
goto v_reusejp_1250_;
}
else
{
lean_object* v_reuseFailAlloc_1252_; 
v_reuseFailAlloc_1252_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1252_, 0, v_a_1246_);
v___x_1251_ = v_reuseFailAlloc_1252_;
goto v_reusejp_1250_;
}
v_reusejp_1250_:
{
return v___x_1251_;
}
}
}
}
else
{
lean_object* v___x_1254_; lean_object* v___x_1255_; lean_object* v___x_1256_; uint8_t v___x_1257_; 
v___x_1254_ = lean_unsigned_to_nat(0u);
v___x_1255_ = l_Lean_Syntax_getArg(v_a_1239_, v___x_1254_);
v___x_1256_ = ((lean_object*)(lp_aesop_Aesop_Frontend_Parser_additionalRule___closed__6));
lean_inc(v___x_1255_);
v___x_1257_ = l_Lean_Syntax_isOfKind(v___x_1255_, v___x_1256_);
if (v___x_1257_ == 0)
{
lean_object* v___x_1258_; 
v___x_1258_ = lp_aesop_Aesop_Frontend_elabForwardRule(v___x_1255_, v___y_1219_, v___y_1220_, v___y_1221_, v___y_1222_, v___y_1223_, v___y_1224_, v___y_1225_);
if (lean_obj_tag(v___x_1258_) == 0)
{
lean_object* v_a_1259_; lean_object* v___x_1260_; lean_object* v___x_1262_; 
v_a_1259_ = lean_ctor_get(v___x_1258_, 0);
lean_inc(v_a_1259_);
lean_dec_ref_known(v___x_1258_, 1);
v___x_1260_ = lp_aesop_Aesop_LocalRuleSet_add(v_fst_1234_, v_a_1259_);
if (v_isShared_1238_ == 0)
{
lean_ctor_set(v___x_1237_, 0, v___x_1260_);
v___x_1262_ = v___x_1237_;
goto v_reusejp_1261_;
}
else
{
lean_object* v_reuseFailAlloc_1263_; 
v_reuseFailAlloc_1263_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1263_, 0, v___x_1260_);
lean_ctor_set(v_reuseFailAlloc_1263_, 1, v_snd_1235_);
v___x_1262_ = v_reuseFailAlloc_1263_;
goto v_reusejp_1261_;
}
v_reusejp_1261_:
{
v_a_1228_ = v___x_1262_;
goto v___jp_1227_;
}
}
else
{
lean_object* v_a_1264_; lean_object* v___x_1266_; uint8_t v_isShared_1267_; uint8_t v_isSharedCheck_1271_; 
lean_del_object(v___x_1237_);
lean_dec(v_snd_1235_);
lean_dec(v_fst_1234_);
v_a_1264_ = lean_ctor_get(v___x_1258_, 0);
v_isSharedCheck_1271_ = !lean_is_exclusive(v___x_1258_);
if (v_isSharedCheck_1271_ == 0)
{
v___x_1266_ = v___x_1258_;
v_isShared_1267_ = v_isSharedCheck_1271_;
goto v_resetjp_1265_;
}
else
{
lean_inc(v_a_1264_);
lean_dec(v___x_1258_);
v___x_1266_ = lean_box(0);
v_isShared_1267_ = v_isSharedCheck_1271_;
goto v_resetjp_1265_;
}
v_resetjp_1265_:
{
lean_object* v___x_1269_; 
if (v_isShared_1267_ == 0)
{
v___x_1269_ = v___x_1266_;
goto v_reusejp_1268_;
}
else
{
lean_object* v_reuseFailAlloc_1270_; 
v_reuseFailAlloc_1270_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1270_, 0, v_a_1264_);
v___x_1269_ = v_reuseFailAlloc_1270_;
goto v_reusejp_1268_;
}
v_reusejp_1268_:
{
return v___x_1269_;
}
}
}
}
else
{
lean_object* v___x_1272_; lean_object* v___x_1274_; 
lean_dec(v___x_1255_);
lean_dec(v_snd_1235_);
v___x_1272_ = lean_box(v___x_1257_);
if (v_isShared_1238_ == 0)
{
lean_ctor_set(v___x_1237_, 1, v___x_1272_);
v___x_1274_ = v___x_1237_;
goto v_reusejp_1273_;
}
else
{
lean_object* v_reuseFailAlloc_1275_; 
v_reuseFailAlloc_1275_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1275_, 0, v_fst_1234_);
lean_ctor_set(v_reuseFailAlloc_1275_, 1, v___x_1272_);
v___x_1274_ = v_reuseFailAlloc_1275_;
goto v_reusejp_1273_;
}
v_reusejp_1273_:
{
v_a_1228_ = v___x_1274_;
goto v___jp_1227_;
}
}
}
}
}
v___jp_1227_:
{
size_t v___x_1229_; size_t v___x_1230_; 
v___x_1229_ = ((size_t)1ULL);
v___x_1230_ = lean_usize_add(v_i_1217_, v___x_1229_);
v_i_1217_ = v___x_1230_;
v_b_1218_ = v_a_1228_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Frontend_elabAdditionalForwardRules_spec__1___boxed(lean_object* v_as_1277_, lean_object* v_sz_1278_, lean_object* v_i_1279_, lean_object* v_b_1280_, lean_object* v___y_1281_, lean_object* v___y_1282_, lean_object* v___y_1283_, lean_object* v___y_1284_, lean_object* v___y_1285_, lean_object* v___y_1286_, lean_object* v___y_1287_, lean_object* v___y_1288_){
_start:
{
size_t v_sz_boxed_1289_; size_t v_i_boxed_1290_; lean_object* v_res_1291_; 
v_sz_boxed_1289_ = lean_unbox_usize(v_sz_1278_);
lean_dec(v_sz_1278_);
v_i_boxed_1290_ = lean_unbox_usize(v_i_1279_);
lean_dec(v_i_1279_);
v_res_1291_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Frontend_elabAdditionalForwardRules_spec__1(v_as_1277_, v_sz_boxed_1289_, v_i_boxed_1290_, v_b_1280_, v___y_1281_, v___y_1282_, v___y_1283_, v___y_1284_, v___y_1285_, v___y_1286_, v___y_1287_);
lean_dec(v___y_1287_);
lean_dec_ref(v___y_1286_);
lean_dec(v___y_1285_);
lean_dec_ref(v___y_1284_);
lean_dec(v___y_1283_);
lean_dec_ref(v___y_1282_);
lean_dec_ref(v___y_1281_);
lean_dec_ref(v_as_1277_);
return v_res_1291_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_elabAdditionalForwardRules(lean_object* v_rs_1292_, lean_object* v_rules_1293_, lean_object* v_a_1294_, lean_object* v_a_1295_, lean_object* v_a_1296_, lean_object* v_a_1297_, lean_object* v_a_1298_, lean_object* v_a_1299_, lean_object* v_a_1300_){
_start:
{
uint8_t v_addImplications_1302_; lean_object* v___x_1303_; lean_object* v___x_1304_; size_t v_sz_1305_; size_t v___x_1306_; lean_object* v___x_1307_; 
v_addImplications_1302_ = 0;
v___x_1303_ = lean_box(v_addImplications_1302_);
v___x_1304_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1304_, 0, v_rs_1292_);
lean_ctor_set(v___x_1304_, 1, v___x_1303_);
v_sz_1305_ = lean_array_size(v_rules_1293_);
v___x_1306_ = ((size_t)0ULL);
v___x_1307_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Frontend_elabAdditionalForwardRules_spec__1(v_rules_1293_, v_sz_1305_, v___x_1306_, v___x_1304_, v_a_1294_, v_a_1295_, v_a_1296_, v_a_1297_, v_a_1298_, v_a_1299_, v_a_1300_);
if (lean_obj_tag(v___x_1307_) == 0)
{
lean_object* v_a_1308_; lean_object* v___x_1310_; uint8_t v_isShared_1311_; uint8_t v_isSharedCheck_1320_; 
v_a_1308_ = lean_ctor_get(v___x_1307_, 0);
v_isSharedCheck_1320_ = !lean_is_exclusive(v___x_1307_);
if (v_isSharedCheck_1320_ == 0)
{
v___x_1310_ = v___x_1307_;
v_isShared_1311_ = v_isSharedCheck_1320_;
goto v_resetjp_1309_;
}
else
{
lean_inc(v_a_1308_);
lean_dec(v___x_1307_);
v___x_1310_ = lean_box(0);
v_isShared_1311_ = v_isSharedCheck_1320_;
goto v_resetjp_1309_;
}
v_resetjp_1309_:
{
lean_object* v_snd_1312_; uint8_t v___x_1313_; 
v_snd_1312_ = lean_ctor_get(v_a_1308_, 1);
v___x_1313_ = lean_unbox(v_snd_1312_);
if (v___x_1313_ == 0)
{
lean_object* v_fst_1314_; lean_object* v___x_1316_; 
v_fst_1314_ = lean_ctor_get(v_a_1308_, 0);
lean_inc(v_fst_1314_);
lean_dec(v_a_1308_);
if (v_isShared_1311_ == 0)
{
lean_ctor_set(v___x_1310_, 0, v_fst_1314_);
v___x_1316_ = v___x_1310_;
goto v_reusejp_1315_;
}
else
{
lean_object* v_reuseFailAlloc_1317_; 
v_reuseFailAlloc_1317_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1317_, 0, v_fst_1314_);
v___x_1316_ = v_reuseFailAlloc_1317_;
goto v_reusejp_1315_;
}
v_reusejp_1315_:
{
return v___x_1316_;
}
}
else
{
lean_object* v_fst_1318_; lean_object* v___x_1319_; 
lean_del_object(v___x_1310_);
v_fst_1318_ = lean_ctor_get(v_a_1308_, 0);
lean_inc(v_fst_1318_);
lean_dec(v_a_1308_);
v___x_1319_ = lp_aesop_Aesop_Frontend_addLocalImplications(v_fst_1318_, v_a_1294_, v_a_1295_, v_a_1296_, v_a_1297_, v_a_1298_, v_a_1299_, v_a_1300_);
return v___x_1319_;
}
}
}
else
{
lean_object* v_a_1321_; lean_object* v___x_1323_; uint8_t v_isShared_1324_; uint8_t v_isSharedCheck_1328_; 
v_a_1321_ = lean_ctor_get(v___x_1307_, 0);
v_isSharedCheck_1328_ = !lean_is_exclusive(v___x_1307_);
if (v_isSharedCheck_1328_ == 0)
{
v___x_1323_ = v___x_1307_;
v_isShared_1324_ = v_isSharedCheck_1328_;
goto v_resetjp_1322_;
}
else
{
lean_inc(v_a_1321_);
lean_dec(v___x_1307_);
v___x_1323_ = lean_box(0);
v_isShared_1324_ = v_isSharedCheck_1328_;
goto v_resetjp_1322_;
}
v_resetjp_1322_:
{
lean_object* v___x_1326_; 
if (v_isShared_1324_ == 0)
{
v___x_1326_ = v___x_1323_;
goto v_reusejp_1325_;
}
else
{
lean_object* v_reuseFailAlloc_1327_; 
v_reuseFailAlloc_1327_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1327_, 0, v_a_1321_);
v___x_1326_ = v_reuseFailAlloc_1327_;
goto v_reusejp_1325_;
}
v_reusejp_1325_:
{
return v___x_1326_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_elabAdditionalForwardRules___boxed(lean_object* v_rs_1329_, lean_object* v_rules_1330_, lean_object* v_a_1331_, lean_object* v_a_1332_, lean_object* v_a_1333_, lean_object* v_a_1334_, lean_object* v_a_1335_, lean_object* v_a_1336_, lean_object* v_a_1337_, lean_object* v_a_1338_){
_start:
{
lean_object* v_res_1339_; 
v_res_1339_ = lp_aesop_Aesop_Frontend_elabAdditionalForwardRules(v_rs_1329_, v_rules_1330_, v_a_1331_, v_a_1332_, v_a_1333_, v_a_1334_, v_a_1335_, v_a_1336_, v_a_1337_);
lean_dec(v_a_1337_);
lean_dec_ref(v_a_1336_);
lean_dec(v_a_1335_);
lean_dec_ref(v_a_1334_);
lean_dec(v_a_1333_);
lean_dec_ref(v_a_1332_);
lean_dec_ref(v_a_1331_);
lean_dec_ref(v_rules_1330_);
return v_res_1339_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_elabForwardRuleSetCore(lean_object* v_rsNames_1340_, lean_object* v_additionalRules_1341_, lean_object* v_options_1342_, lean_object* v_a_1343_, lean_object* v_a_1344_, lean_object* v_a_1345_, lean_object* v_a_1346_, lean_object* v_a_1347_, lean_object* v_a_1348_, lean_object* v_a_1349_){
_start:
{
lean_object* v___x_1351_; 
v___x_1351_ = lp_aesop_Aesop_Frontend_elabGlobalRuleSets(v_rsNames_1340_, v_a_1348_, v_a_1349_);
if (lean_obj_tag(v___x_1351_) == 0)
{
lean_object* v_a_1352_; lean_object* v___x_1353_; 
v_a_1352_ = lean_ctor_get(v___x_1351_, 0);
lean_inc(v_a_1352_);
lean_dec_ref_known(v___x_1351_, 1);
v___x_1353_ = lp_aesop_Aesop_mkLocalRuleSet(v_a_1352_, v_options_1342_, v_a_1348_, v_a_1349_);
lean_dec(v_a_1352_);
if (lean_obj_tag(v___x_1353_) == 0)
{
lean_object* v_a_1354_; lean_object* v___x_1355_; 
v_a_1354_ = lean_ctor_get(v___x_1353_, 0);
lean_inc(v_a_1354_);
lean_dec_ref_known(v___x_1353_, 1);
v___x_1355_ = lp_aesop_Aesop_Frontend_elabAdditionalForwardRules(v_a_1354_, v_additionalRules_1341_, v_a_1343_, v_a_1344_, v_a_1345_, v_a_1346_, v_a_1347_, v_a_1348_, v_a_1349_);
return v___x_1355_;
}
else
{
return v___x_1353_;
}
}
else
{
lean_object* v_a_1356_; lean_object* v___x_1358_; uint8_t v_isShared_1359_; uint8_t v_isSharedCheck_1363_; 
v_a_1356_ = lean_ctor_get(v___x_1351_, 0);
v_isSharedCheck_1363_ = !lean_is_exclusive(v___x_1351_);
if (v_isSharedCheck_1363_ == 0)
{
v___x_1358_ = v___x_1351_;
v_isShared_1359_ = v_isSharedCheck_1363_;
goto v_resetjp_1357_;
}
else
{
lean_inc(v_a_1356_);
lean_dec(v___x_1351_);
v___x_1358_ = lean_box(0);
v_isShared_1359_ = v_isSharedCheck_1363_;
goto v_resetjp_1357_;
}
v_resetjp_1357_:
{
lean_object* v___x_1361_; 
if (v_isShared_1359_ == 0)
{
v___x_1361_ = v___x_1358_;
goto v_reusejp_1360_;
}
else
{
lean_object* v_reuseFailAlloc_1362_; 
v_reuseFailAlloc_1362_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1362_, 0, v_a_1356_);
v___x_1361_ = v_reuseFailAlloc_1362_;
goto v_reusejp_1360_;
}
v_reusejp_1360_:
{
return v___x_1361_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_elabForwardRuleSetCore___boxed(lean_object* v_rsNames_1364_, lean_object* v_additionalRules_1365_, lean_object* v_options_1366_, lean_object* v_a_1367_, lean_object* v_a_1368_, lean_object* v_a_1369_, lean_object* v_a_1370_, lean_object* v_a_1371_, lean_object* v_a_1372_, lean_object* v_a_1373_, lean_object* v_a_1374_){
_start:
{
lean_object* v_res_1375_; 
v_res_1375_ = lp_aesop_Aesop_Frontend_elabForwardRuleSetCore(v_rsNames_1364_, v_additionalRules_1365_, v_options_1366_, v_a_1367_, v_a_1368_, v_a_1369_, v_a_1370_, v_a_1371_, v_a_1372_, v_a_1373_);
lean_dec(v_a_1373_);
lean_dec_ref(v_a_1372_);
lean_dec(v_a_1371_);
lean_dec_ref(v_a_1370_);
lean_dec(v_a_1369_);
lean_dec_ref(v_a_1368_);
lean_dec_ref(v_a_1367_);
lean_dec_ref(v_options_1366_);
lean_dec_ref(v_additionalRules_1365_);
return v_res_1375_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_elabForwardRuleSet(lean_object* v_rsNames_x3f_1378_, lean_object* v_additionalRules_x3f_1379_, lean_object* v_options_1380_, lean_object* v_a_1381_, lean_object* v_a_1382_, lean_object* v_a_1383_, lean_object* v_a_1384_, lean_object* v_a_1385_, lean_object* v_a_1386_, lean_object* v_a_1387_){
_start:
{
lean_object* v___y_1390_; lean_object* v___y_1391_; lean_object* v___y_1397_; lean_object* v_a_1398_; lean_object* v___y_1402_; lean_object* v_a_1403_; lean_object* v_a_1406_; lean_object* v_a_1425_; 
if (lean_obj_tag(v_rsNames_x3f_1378_) == 0)
{
lean_object* v___x_1427_; 
v___x_1427_ = lean_box(0);
v_a_1406_ = v___x_1427_;
goto v___jp_1405_;
}
else
{
lean_object* v_val_1428_; lean_object* v___x_1429_; uint8_t v___x_1430_; 
v_val_1428_ = lean_ctor_get(v_rsNames_x3f_1378_, 0);
lean_inc_n(v_val_1428_, 2);
lean_dec_ref_known(v_rsNames_x3f_1378_, 1);
v___x_1429_ = ((lean_object*)(lp_aesop_Aesop_Frontend_Parser_usingRuleSets___closed__4));
v___x_1430_ = l_Lean_Syntax_isOfKind(v_val_1428_, v___x_1429_);
if (v___x_1430_ == 0)
{
lean_object* v___x_1431_; lean_object* v_a_1432_; lean_object* v___x_1434_; uint8_t v_isShared_1435_; uint8_t v_isSharedCheck_1439_; 
lean_dec(v_val_1428_);
lean_dec(v_additionalRules_x3f_1379_);
v___x_1431_ = lp_aesop_Lean_Elab_throwUnsupportedSyntax___at___00Aesop_Frontend_elabAdditionalForwardRules_spec__0___redArg();
v_a_1432_ = lean_ctor_get(v___x_1431_, 0);
v_isSharedCheck_1439_ = !lean_is_exclusive(v___x_1431_);
if (v_isSharedCheck_1439_ == 0)
{
v___x_1434_ = v___x_1431_;
v_isShared_1435_ = v_isSharedCheck_1439_;
goto v_resetjp_1433_;
}
else
{
lean_inc(v_a_1432_);
lean_dec(v___x_1431_);
v___x_1434_ = lean_box(0);
v_isShared_1435_ = v_isSharedCheck_1439_;
goto v_resetjp_1433_;
}
v_resetjp_1433_:
{
lean_object* v___x_1437_; 
if (v_isShared_1435_ == 0)
{
v___x_1437_ = v___x_1434_;
goto v_reusejp_1436_;
}
else
{
lean_object* v_reuseFailAlloc_1438_; 
v_reuseFailAlloc_1438_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1438_, 0, v_a_1432_);
v___x_1437_ = v_reuseFailAlloc_1438_;
goto v_reusejp_1436_;
}
v_reusejp_1436_:
{
return v___x_1437_;
}
}
}
else
{
lean_object* v___x_1440_; lean_object* v___x_1441_; lean_object* v_rs_1442_; 
v___x_1440_ = lean_unsigned_to_nat(1u);
v___x_1441_ = l_Lean_Syntax_getArg(v_val_1428_, v___x_1440_);
lean_dec(v_val_1428_);
v_rs_1442_ = l_Lean_Syntax_getArgs(v___x_1441_);
lean_dec(v___x_1441_);
v_a_1425_ = v_rs_1442_;
goto v___jp_1424_;
}
}
v___jp_1389_:
{
if (lean_obj_tag(v___y_1390_) == 0)
{
lean_object* v___x_1392_; lean_object* v___x_1393_; 
v___x_1392_ = ((lean_object*)(lp_aesop_Aesop_Frontend_elabForwardRuleSet___closed__0));
v___x_1393_ = lp_aesop_Aesop_Frontend_elabForwardRuleSetCore(v___y_1391_, v___x_1392_, v_options_1380_, v_a_1381_, v_a_1382_, v_a_1383_, v_a_1384_, v_a_1385_, v_a_1386_, v_a_1387_);
return v___x_1393_;
}
else
{
lean_object* v_val_1394_; lean_object* v___x_1395_; 
v_val_1394_ = lean_ctor_get(v___y_1390_, 0);
lean_inc(v_val_1394_);
lean_dec_ref_known(v___y_1390_, 1);
v___x_1395_ = lp_aesop_Aesop_Frontend_elabForwardRuleSetCore(v___y_1391_, v_val_1394_, v_options_1380_, v_a_1381_, v_a_1382_, v_a_1383_, v_a_1384_, v_a_1385_, v_a_1386_, v_a_1387_);
lean_dec(v_val_1394_);
return v___x_1395_;
}
}
v___jp_1396_:
{
if (lean_obj_tag(v___y_1397_) == 0)
{
lean_object* v___x_1399_; 
v___x_1399_ = ((lean_object*)(lp_aesop_Aesop_Frontend_elabForwardRuleSet___closed__0));
v___y_1390_ = v_a_1398_;
v___y_1391_ = v___x_1399_;
goto v___jp_1389_;
}
else
{
lean_object* v_val_1400_; 
v_val_1400_ = lean_ctor_get(v___y_1397_, 0);
lean_inc(v_val_1400_);
lean_dec_ref_known(v___y_1397_, 1);
v___y_1390_ = v_a_1398_;
v___y_1391_ = v_val_1400_;
goto v___jp_1389_;
}
}
v___jp_1401_:
{
lean_object* v___x_1404_; 
v___x_1404_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1404_, 0, v_a_1403_);
v___y_1397_ = v___y_1402_;
v_a_1398_ = v___x_1404_;
goto v___jp_1396_;
}
v___jp_1405_:
{
if (lean_obj_tag(v_additionalRules_x3f_1379_) == 0)
{
lean_object* v___x_1407_; 
v___x_1407_ = lean_box(0);
v___y_1397_ = v_a_1406_;
v_a_1398_ = v___x_1407_;
goto v___jp_1396_;
}
else
{
lean_object* v_val_1408_; lean_object* v___x_1409_; uint8_t v___x_1410_; 
v_val_1408_ = lean_ctor_get(v_additionalRules_x3f_1379_, 0);
lean_inc_n(v_val_1408_, 2);
lean_dec_ref_known(v_additionalRules_x3f_1379_, 1);
v___x_1409_ = ((lean_object*)(lp_aesop_Aesop_Frontend_Parser_additionalRules___closed__1));
v___x_1410_ = l_Lean_Syntax_isOfKind(v_val_1408_, v___x_1409_);
if (v___x_1410_ == 0)
{
lean_object* v___x_1411_; lean_object* v_a_1412_; lean_object* v___x_1414_; uint8_t v_isShared_1415_; uint8_t v_isSharedCheck_1419_; 
lean_dec(v_val_1408_);
lean_dec(v_a_1406_);
v___x_1411_ = lp_aesop_Lean_Elab_throwUnsupportedSyntax___at___00Aesop_Frontend_elabAdditionalForwardRules_spec__0___redArg();
v_a_1412_ = lean_ctor_get(v___x_1411_, 0);
v_isSharedCheck_1419_ = !lean_is_exclusive(v___x_1411_);
if (v_isSharedCheck_1419_ == 0)
{
v___x_1414_ = v___x_1411_;
v_isShared_1415_ = v_isSharedCheck_1419_;
goto v_resetjp_1413_;
}
else
{
lean_inc(v_a_1412_);
lean_dec(v___x_1411_);
v___x_1414_ = lean_box(0);
v_isShared_1415_ = v_isSharedCheck_1419_;
goto v_resetjp_1413_;
}
v_resetjp_1413_:
{
lean_object* v___x_1417_; 
if (v_isShared_1415_ == 0)
{
v___x_1417_ = v___x_1414_;
goto v_reusejp_1416_;
}
else
{
lean_object* v_reuseFailAlloc_1418_; 
v_reuseFailAlloc_1418_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1418_, 0, v_a_1412_);
v___x_1417_ = v_reuseFailAlloc_1418_;
goto v_reusejp_1416_;
}
v_reusejp_1416_:
{
return v___x_1417_;
}
}
}
else
{
lean_object* v___x_1420_; lean_object* v___x_1421_; lean_object* v___x_1422_; lean_object* v___x_1423_; 
v___x_1420_ = lean_unsigned_to_nat(1u);
v___x_1421_ = l_Lean_Syntax_getArg(v_val_1408_, v___x_1420_);
lean_dec(v_val_1408_);
v___x_1422_ = l_Lean_Syntax_getArgs(v___x_1421_);
lean_dec(v___x_1421_);
v___x_1423_ = l_Lean_Syntax_TSepArray_getElems___redArg(v___x_1422_);
lean_dec_ref(v___x_1422_);
v___y_1402_ = v_a_1406_;
v_a_1403_ = v___x_1423_;
goto v___jp_1401_;
}
}
}
v___jp_1424_:
{
lean_object* v___x_1426_; 
v___x_1426_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1426_, 0, v_a_1425_);
v_a_1406_ = v___x_1426_;
goto v___jp_1405_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_elabForwardRuleSet___boxed(lean_object* v_rsNames_x3f_1443_, lean_object* v_additionalRules_x3f_1444_, lean_object* v_options_1445_, lean_object* v_a_1446_, lean_object* v_a_1447_, lean_object* v_a_1448_, lean_object* v_a_1449_, lean_object* v_a_1450_, lean_object* v_a_1451_, lean_object* v_a_1452_, lean_object* v_a_1453_){
_start:
{
lean_object* v_res_1454_; 
v_res_1454_ = lp_aesop_Aesop_Frontend_elabForwardRuleSet(v_rsNames_x3f_1443_, v_additionalRules_x3f_1444_, v_options_1445_, v_a_1446_, v_a_1447_, v_a_1448_, v_a_1449_, v_a_1450_, v_a_1451_, v_a_1452_);
lean_dec(v_a_1452_);
lean_dec_ref(v_a_1451_);
lean_dec(v_a_1450_);
lean_dec_ref(v_a_1449_);
lean_dec(v_a_1448_);
lean_dec_ref(v_a_1447_);
lean_dec_ref(v_a_1446_);
lean_dec_ref(v_options_1445_);
return v_res_1454_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_evalSaturateCore___lam__0(lean_object* v_a_1455_, lean_object* v_a_1456_, lean_object* v___y_1457_, lean_object* v___y_1458_, lean_object* v___y_1459_, lean_object* v___y_1460_, lean_object* v___y_1461_, lean_object* v___y_1462_, lean_object* v___y_1463_, lean_object* v___y_1464_){
_start:
{
lean_object* v___x_1466_; 
v___x_1466_ = l_Lean_Elab_Tactic_getMainGoal___redArg(v___y_1458_, v___y_1461_, v___y_1462_, v___y_1463_, v___y_1464_);
if (lean_obj_tag(v___x_1466_) == 0)
{
lean_object* v_a_1467_; lean_object* v___x_1468_; 
v_a_1467_ = lean_ctor_get(v___x_1466_, 0);
lean_inc(v_a_1467_);
lean_dec_ref_known(v___x_1466_, 1);
v___x_1468_ = lp_aesop_Aesop_saturate(v_a_1455_, v_a_1467_, v_a_1456_, v___y_1461_, v___y_1462_, v___y_1463_, v___y_1464_);
if (lean_obj_tag(v___x_1468_) == 0)
{
lean_object* v_a_1469_; lean_object* v_fst_1470_; lean_object* v_snd_1471_; lean_object* v___x_1473_; uint8_t v_isShared_1474_; uint8_t v_isSharedCheck_1496_; 
v_a_1469_ = lean_ctor_get(v___x_1468_, 0);
lean_inc(v_a_1469_);
lean_dec_ref_known(v___x_1468_, 1);
v_fst_1470_ = lean_ctor_get(v_a_1469_, 0);
v_snd_1471_ = lean_ctor_get(v_a_1469_, 1);
v_isSharedCheck_1496_ = !lean_is_exclusive(v_a_1469_);
if (v_isSharedCheck_1496_ == 0)
{
v___x_1473_ = v_a_1469_;
v_isShared_1474_ = v_isSharedCheck_1496_;
goto v_resetjp_1472_;
}
else
{
lean_inc(v_snd_1471_);
lean_inc(v_fst_1470_);
lean_dec(v_a_1469_);
v___x_1473_ = lean_box(0);
v_isShared_1474_ = v_isSharedCheck_1496_;
goto v_resetjp_1472_;
}
v_resetjp_1472_:
{
lean_object* v___x_1475_; lean_object* v___x_1477_; 
v___x_1475_ = lean_box(0);
if (v_isShared_1474_ == 0)
{
lean_ctor_set_tag(v___x_1473_, 1);
lean_ctor_set(v___x_1473_, 1, v___x_1475_);
v___x_1477_ = v___x_1473_;
goto v_reusejp_1476_;
}
else
{
lean_object* v_reuseFailAlloc_1495_; 
v_reuseFailAlloc_1495_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1495_, 0, v_fst_1470_);
lean_ctor_set(v_reuseFailAlloc_1495_, 1, v___x_1475_);
v___x_1477_ = v_reuseFailAlloc_1495_;
goto v_reusejp_1476_;
}
v_reusejp_1476_:
{
lean_object* v___x_1478_; 
v___x_1478_ = l_Lean_Elab_Tactic_replaceMainGoal___redArg(v___x_1477_, v___y_1458_, v___y_1461_, v___y_1462_, v___y_1463_, v___y_1464_);
if (lean_obj_tag(v___x_1478_) == 0)
{
lean_object* v___x_1480_; uint8_t v_isShared_1481_; uint8_t v_isSharedCheck_1485_; 
v_isSharedCheck_1485_ = !lean_is_exclusive(v___x_1478_);
if (v_isSharedCheck_1485_ == 0)
{
lean_object* v_unused_1486_; 
v_unused_1486_ = lean_ctor_get(v___x_1478_, 0);
lean_dec(v_unused_1486_);
v___x_1480_ = v___x_1478_;
v_isShared_1481_ = v_isSharedCheck_1485_;
goto v_resetjp_1479_;
}
else
{
lean_dec(v___x_1478_);
v___x_1480_ = lean_box(0);
v_isShared_1481_ = v_isSharedCheck_1485_;
goto v_resetjp_1479_;
}
v_resetjp_1479_:
{
lean_object* v___x_1483_; 
if (v_isShared_1481_ == 0)
{
lean_ctor_set(v___x_1480_, 0, v_snd_1471_);
v___x_1483_ = v___x_1480_;
goto v_reusejp_1482_;
}
else
{
lean_object* v_reuseFailAlloc_1484_; 
v_reuseFailAlloc_1484_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1484_, 0, v_snd_1471_);
v___x_1483_ = v_reuseFailAlloc_1484_;
goto v_reusejp_1482_;
}
v_reusejp_1482_:
{
return v___x_1483_;
}
}
}
else
{
lean_object* v_a_1487_; lean_object* v___x_1489_; uint8_t v_isShared_1490_; uint8_t v_isSharedCheck_1494_; 
lean_dec(v_snd_1471_);
v_a_1487_ = lean_ctor_get(v___x_1478_, 0);
v_isSharedCheck_1494_ = !lean_is_exclusive(v___x_1478_);
if (v_isSharedCheck_1494_ == 0)
{
v___x_1489_ = v___x_1478_;
v_isShared_1490_ = v_isSharedCheck_1494_;
goto v_resetjp_1488_;
}
else
{
lean_inc(v_a_1487_);
lean_dec(v___x_1478_);
v___x_1489_ = lean_box(0);
v_isShared_1490_ = v_isSharedCheck_1494_;
goto v_resetjp_1488_;
}
v_resetjp_1488_:
{
lean_object* v___x_1492_; 
if (v_isShared_1490_ == 0)
{
v___x_1492_ = v___x_1489_;
goto v_reusejp_1491_;
}
else
{
lean_object* v_reuseFailAlloc_1493_; 
v_reuseFailAlloc_1493_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1493_, 0, v_a_1487_);
v___x_1492_ = v_reuseFailAlloc_1493_;
goto v_reusejp_1491_;
}
v_reusejp_1491_:
{
return v___x_1492_;
}
}
}
}
}
}
else
{
lean_object* v_a_1497_; lean_object* v___x_1499_; uint8_t v_isShared_1500_; uint8_t v_isSharedCheck_1504_; 
v_a_1497_ = lean_ctor_get(v___x_1468_, 0);
v_isSharedCheck_1504_ = !lean_is_exclusive(v___x_1468_);
if (v_isSharedCheck_1504_ == 0)
{
v___x_1499_ = v___x_1468_;
v_isShared_1500_ = v_isSharedCheck_1504_;
goto v_resetjp_1498_;
}
else
{
lean_inc(v_a_1497_);
lean_dec(v___x_1468_);
v___x_1499_ = lean_box(0);
v_isShared_1500_ = v_isSharedCheck_1504_;
goto v_resetjp_1498_;
}
v_resetjp_1498_:
{
lean_object* v___x_1502_; 
if (v_isShared_1500_ == 0)
{
v___x_1502_ = v___x_1499_;
goto v_reusejp_1501_;
}
else
{
lean_object* v_reuseFailAlloc_1503_; 
v_reuseFailAlloc_1503_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1503_, 0, v_a_1497_);
v___x_1502_ = v_reuseFailAlloc_1503_;
goto v_reusejp_1501_;
}
v_reusejp_1501_:
{
return v___x_1502_;
}
}
}
}
else
{
lean_object* v_a_1505_; lean_object* v___x_1507_; uint8_t v_isShared_1508_; uint8_t v_isSharedCheck_1512_; 
lean_dec_ref(v_a_1456_);
lean_dec_ref(v_a_1455_);
v_a_1505_ = lean_ctor_get(v___x_1466_, 0);
v_isSharedCheck_1512_ = !lean_is_exclusive(v___x_1466_);
if (v_isSharedCheck_1512_ == 0)
{
v___x_1507_ = v___x_1466_;
v_isShared_1508_ = v_isSharedCheck_1512_;
goto v_resetjp_1506_;
}
else
{
lean_inc(v_a_1505_);
lean_dec(v___x_1466_);
v___x_1507_ = lean_box(0);
v_isShared_1508_ = v_isSharedCheck_1512_;
goto v_resetjp_1506_;
}
v_resetjp_1506_:
{
lean_object* v___x_1510_; 
if (v_isShared_1508_ == 0)
{
v___x_1510_ = v___x_1507_;
goto v_reusejp_1509_;
}
else
{
lean_object* v_reuseFailAlloc_1511_; 
v_reuseFailAlloc_1511_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1511_, 0, v_a_1505_);
v___x_1510_ = v_reuseFailAlloc_1511_;
goto v_reusejp_1509_;
}
v_reusejp_1509_:
{
return v___x_1510_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_evalSaturateCore___lam__0___boxed(lean_object* v_a_1513_, lean_object* v_a_1514_, lean_object* v___y_1515_, lean_object* v___y_1516_, lean_object* v___y_1517_, lean_object* v___y_1518_, lean_object* v___y_1519_, lean_object* v___y_1520_, lean_object* v___y_1521_, lean_object* v___y_1522_, lean_object* v___y_1523_){
_start:
{
lean_object* v_res_1524_; 
v_res_1524_ = lp_aesop_Aesop_Frontend_evalSaturateCore___lam__0(v_a_1513_, v_a_1514_, v___y_1515_, v___y_1516_, v___y_1517_, v___y_1518_, v___y_1519_, v___y_1520_, v___y_1521_, v___y_1522_);
lean_dec(v___y_1522_);
lean_dec_ref(v___y_1521_);
lean_dec(v___y_1520_);
lean_dec_ref(v___y_1519_);
lean_dec(v___y_1518_);
lean_dec_ref(v___y_1517_);
lean_dec(v___y_1516_);
lean_dec_ref(v___y_1515_);
return v_res_1524_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_get___at___00Aesop_appendStatsToStatsFileIfEnabled___at___00Aesop_Frontend_evalSaturateCore_spec__0_spec__0(lean_object* v_opts_1525_, lean_object* v_opt_1526_){
_start:
{
lean_object* v_name_1527_; lean_object* v_defValue_1528_; lean_object* v_map_1529_; lean_object* v___x_1530_; 
v_name_1527_ = lean_ctor_get(v_opt_1526_, 0);
v_defValue_1528_ = lean_ctor_get(v_opt_1526_, 1);
v_map_1529_ = lean_ctor_get(v_opts_1525_, 0);
v___x_1530_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_1529_, v_name_1527_);
if (lean_obj_tag(v___x_1530_) == 0)
{
lean_inc(v_defValue_1528_);
return v_defValue_1528_;
}
else
{
lean_object* v_val_1531_; 
v_val_1531_ = lean_ctor_get(v___x_1530_, 0);
lean_inc(v_val_1531_);
lean_dec_ref_known(v___x_1530_, 1);
if (lean_obj_tag(v_val_1531_) == 0)
{
lean_object* v_v_1532_; 
v_v_1532_ = lean_ctor_get(v_val_1531_, 0);
lean_inc_ref(v_v_1532_);
lean_dec_ref_known(v_val_1531_, 1);
return v_v_1532_;
}
else
{
lean_dec(v_val_1531_);
lean_inc(v_defValue_1528_);
return v_defValue_1528_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_get___at___00Aesop_appendStatsToStatsFileIfEnabled___at___00Aesop_Frontend_evalSaturateCore_spec__0_spec__0___boxed(lean_object* v_opts_1533_, lean_object* v_opt_1534_){
_start:
{
lean_object* v_res_1535_; 
v_res_1535_ = lp_aesop_Lean_Option_get___at___00Aesop_appendStatsToStatsFileIfEnabled___at___00Aesop_Frontend_evalSaturateCore_spec__0_spec__0(v_opts_1533_, v_opt_1534_);
lean_dec_ref(v_opt_1534_);
lean_dec_ref(v_opts_1533_);
return v_res_1535_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_StatsFileRecord_ofStats___at___00Aesop_appendStatsToStatsFileIfEnabled___at___00Aesop_Frontend_evalSaturateCore_spec__0_spec__1___redArg(lean_object* v_aesopStx_1539_, uint8_t v_goalSolved_1540_, lean_object* v_stats_1541_, lean_object* v___y_1542_, lean_object* v___y_1543_, lean_object* v___y_1544_){
_start:
{
lean_object* v_fileName_1546_; lean_object* v_fileMap_1547_; lean_object* v___y_1549_; uint8_t v___x_1582_; lean_object* v___x_1583_; 
v_fileName_1546_ = lean_ctor_get(v___y_1543_, 0);
v_fileMap_1547_ = lean_ctor_get(v___y_1543_, 1);
v___x_1582_ = 0;
v___x_1583_ = l_Lean_Syntax_getPos_x3f(v_aesopStx_1539_, v___x_1582_);
if (lean_obj_tag(v___x_1583_) == 0)
{
lean_object* v___x_1584_; 
v___x_1584_ = lean_box(0);
v___y_1549_ = v___x_1584_;
goto v___jp_1548_;
}
else
{
lean_object* v_val_1585_; lean_object* v___x_1587_; uint8_t v_isShared_1588_; uint8_t v_isSharedCheck_1593_; 
v_val_1585_ = lean_ctor_get(v___x_1583_, 0);
v_isSharedCheck_1593_ = !lean_is_exclusive(v___x_1583_);
if (v_isSharedCheck_1593_ == 0)
{
v___x_1587_ = v___x_1583_;
v_isShared_1588_ = v_isSharedCheck_1593_;
goto v_resetjp_1586_;
}
else
{
lean_inc(v_val_1585_);
lean_dec(v___x_1583_);
v___x_1587_ = lean_box(0);
v_isShared_1588_ = v_isSharedCheck_1593_;
goto v_resetjp_1586_;
}
v_resetjp_1586_:
{
lean_object* v___x_1589_; lean_object* v___x_1591_; 
lean_inc_ref(v_fileMap_1547_);
v___x_1589_ = l_Lean_FileMap_toPosition(v_fileMap_1547_, v_val_1585_);
lean_dec(v_val_1585_);
if (v_isShared_1588_ == 0)
{
lean_ctor_set(v___x_1587_, 0, v___x_1589_);
v___x_1591_ = v___x_1587_;
goto v_reusejp_1590_;
}
else
{
lean_object* v_reuseFailAlloc_1592_; 
v_reuseFailAlloc_1592_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1592_, 0, v___x_1589_);
v___x_1591_ = v_reuseFailAlloc_1592_;
goto v_reusejp_1590_;
}
v_reusejp_1590_:
{
v___y_1549_ = v___x_1591_;
goto v___jp_1548_;
}
}
}
v___jp_1548_:
{
lean_object* v___x_1550_; 
v___x_1550_ = l_Lean_Elab_Term_getDeclName_x3f___redArg(v___y_1542_);
if (lean_obj_tag(v___x_1550_) == 0)
{
lean_object* v_a_1551_; lean_object* v___x_1552_; lean_object* v___x_1553_; 
v_a_1551_ = lean_ctor_get(v___x_1550_, 0);
lean_inc(v_a_1551_);
lean_dec_ref_known(v___x_1550_, 1);
v___x_1552_ = ((lean_object*)(lp_aesop_Aesop_StatsFileRecord_ofStats___at___00Aesop_appendStatsToStatsFileIfEnabled___at___00Aesop_Frontend_evalSaturateCore_spec__0_spec__1___redArg___closed__1));
v___x_1553_ = l_Lean_PrettyPrinter_ppCategory(v___x_1552_, v_aesopStx_1539_, v___y_1543_, v___y_1544_);
if (lean_obj_tag(v___x_1553_) == 0)
{
lean_object* v_a_1554_; lean_object* v___x_1556_; uint8_t v_isShared_1557_; uint8_t v_isSharedCheck_1565_; 
v_a_1554_ = lean_ctor_get(v___x_1553_, 0);
v_isSharedCheck_1565_ = !lean_is_exclusive(v___x_1553_);
if (v_isSharedCheck_1565_ == 0)
{
v___x_1556_ = v___x_1553_;
v_isShared_1557_ = v_isSharedCheck_1565_;
goto v_resetjp_1555_;
}
else
{
lean_inc(v_a_1554_);
lean_dec(v___x_1553_);
v___x_1556_ = lean_box(0);
v_isShared_1557_ = v_isSharedCheck_1565_;
goto v_resetjp_1555_;
}
v_resetjp_1555_:
{
lean_object* v___x_1558_; lean_object* v___x_1559_; lean_object* v_syntax_1560_; lean_object* v___x_1561_; lean_object* v___x_1563_; 
v___x_1558_ = lean_cstr_to_nat("100000000000");
v___x_1559_ = lean_unsigned_to_nat(0u);
v_syntax_1560_ = l_Std_Format_pretty(v_a_1554_, v___x_1558_, v___x_1559_, v___x_1559_);
lean_inc_ref(v_fileName_1546_);
v___x_1561_ = lean_alloc_ctor(0, 5, 1);
lean_ctor_set(v___x_1561_, 0, v_stats_1541_);
lean_ctor_set(v___x_1561_, 1, v_syntax_1560_);
lean_ctor_set(v___x_1561_, 2, v_fileName_1546_);
lean_ctor_set(v___x_1561_, 3, v___y_1549_);
lean_ctor_set(v___x_1561_, 4, v_a_1551_);
lean_ctor_set_uint8(v___x_1561_, sizeof(void*)*5, v_goalSolved_1540_);
if (v_isShared_1557_ == 0)
{
lean_ctor_set(v___x_1556_, 0, v___x_1561_);
v___x_1563_ = v___x_1556_;
goto v_reusejp_1562_;
}
else
{
lean_object* v_reuseFailAlloc_1564_; 
v_reuseFailAlloc_1564_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1564_, 0, v___x_1561_);
v___x_1563_ = v_reuseFailAlloc_1564_;
goto v_reusejp_1562_;
}
v_reusejp_1562_:
{
return v___x_1563_;
}
}
}
else
{
lean_object* v_a_1566_; lean_object* v___x_1568_; uint8_t v_isShared_1569_; uint8_t v_isSharedCheck_1573_; 
lean_dec(v_a_1551_);
lean_dec(v___y_1549_);
lean_dec_ref(v_stats_1541_);
v_a_1566_ = lean_ctor_get(v___x_1553_, 0);
v_isSharedCheck_1573_ = !lean_is_exclusive(v___x_1553_);
if (v_isSharedCheck_1573_ == 0)
{
v___x_1568_ = v___x_1553_;
v_isShared_1569_ = v_isSharedCheck_1573_;
goto v_resetjp_1567_;
}
else
{
lean_inc(v_a_1566_);
lean_dec(v___x_1553_);
v___x_1568_ = lean_box(0);
v_isShared_1569_ = v_isSharedCheck_1573_;
goto v_resetjp_1567_;
}
v_resetjp_1567_:
{
lean_object* v___x_1571_; 
if (v_isShared_1569_ == 0)
{
v___x_1571_ = v___x_1568_;
goto v_reusejp_1570_;
}
else
{
lean_object* v_reuseFailAlloc_1572_; 
v_reuseFailAlloc_1572_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1572_, 0, v_a_1566_);
v___x_1571_ = v_reuseFailAlloc_1572_;
goto v_reusejp_1570_;
}
v_reusejp_1570_:
{
return v___x_1571_;
}
}
}
}
else
{
lean_object* v_a_1574_; lean_object* v___x_1576_; uint8_t v_isShared_1577_; uint8_t v_isSharedCheck_1581_; 
lean_dec(v___y_1549_);
lean_dec_ref(v_stats_1541_);
lean_dec(v_aesopStx_1539_);
v_a_1574_ = lean_ctor_get(v___x_1550_, 0);
v_isSharedCheck_1581_ = !lean_is_exclusive(v___x_1550_);
if (v_isSharedCheck_1581_ == 0)
{
v___x_1576_ = v___x_1550_;
v_isShared_1577_ = v_isSharedCheck_1581_;
goto v_resetjp_1575_;
}
else
{
lean_inc(v_a_1574_);
lean_dec(v___x_1550_);
v___x_1576_ = lean_box(0);
v_isShared_1577_ = v_isSharedCheck_1581_;
goto v_resetjp_1575_;
}
v_resetjp_1575_:
{
lean_object* v___x_1579_; 
if (v_isShared_1577_ == 0)
{
v___x_1579_ = v___x_1576_;
goto v_reusejp_1578_;
}
else
{
lean_object* v_reuseFailAlloc_1580_; 
v_reuseFailAlloc_1580_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1580_, 0, v_a_1574_);
v___x_1579_ = v_reuseFailAlloc_1580_;
goto v_reusejp_1578_;
}
v_reusejp_1578_:
{
return v___x_1579_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_StatsFileRecord_ofStats___at___00Aesop_appendStatsToStatsFileIfEnabled___at___00Aesop_Frontend_evalSaturateCore_spec__0_spec__1___redArg___boxed(lean_object* v_aesopStx_1594_, lean_object* v_goalSolved_1595_, lean_object* v_stats_1596_, lean_object* v___y_1597_, lean_object* v___y_1598_, lean_object* v___y_1599_, lean_object* v___y_1600_){
_start:
{
uint8_t v_goalSolved_boxed_1601_; lean_object* v_res_1602_; 
v_goalSolved_boxed_1601_ = lean_unbox(v_goalSolved_1595_);
v_res_1602_ = lp_aesop_Aesop_StatsFileRecord_ofStats___at___00Aesop_appendStatsToStatsFileIfEnabled___at___00Aesop_Frontend_evalSaturateCore_spec__0_spec__1___redArg(v_aesopStx_1594_, v_goalSolved_boxed_1601_, v_stats_1596_, v___y_1597_, v___y_1598_, v___y_1599_);
lean_dec(v___y_1599_);
lean_dec_ref(v___y_1598_);
lean_dec_ref(v___y_1597_);
return v_res_1602_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_appendStatsToStatsFileIfEnabled___at___00Aesop_Frontend_evalSaturateCore_spec__0(lean_object* v_aesopStx_1604_, lean_object* v_stats_1605_, uint8_t v_allGoalsSolved_1606_, lean_object* v___y_1607_, lean_object* v___y_1608_, lean_object* v___y_1609_, lean_object* v___y_1610_, lean_object* v___y_1611_, lean_object* v___y_1612_, lean_object* v___y_1613_, lean_object* v___y_1614_){
_start:
{
lean_object* v_options_1616_; lean_object* v_ref_1617_; lean_object* v_a_1619_; lean_object* v___y_1626_; lean_object* v___x_1636_; lean_object* v_file_1637_; lean_object* v___x_1638_; uint8_t v___x_1639_; 
v_options_1616_ = lean_ctor_get(v___y_1613_, 2);
v_ref_1617_ = lean_ctor_get(v___y_1613_, 5);
v___x_1636_ = lp_aesop_Aesop_aesop_stats_file;
v_file_1637_ = lp_aesop_Lean_Option_get___at___00Aesop_appendStatsToStatsFileIfEnabled___at___00Aesop_Frontend_evalSaturateCore_spec__0_spec__0(v_options_1616_, v___x_1636_);
v___x_1638_ = ((lean_object*)(lp_aesop_Aesop_appendStatsToStatsFileIfEnabled___at___00Aesop_Frontend_evalSaturateCore_spec__0___closed__0));
v___x_1639_ = lean_string_dec_eq(v_file_1637_, v___x_1638_);
if (v___x_1639_ == 0)
{
lean_object* v___x_1640_; 
v___x_1640_ = lp_aesop_Aesop_StatsFileRecord_ofStats___at___00Aesop_appendStatsToStatsFileIfEnabled___at___00Aesop_Frontend_evalSaturateCore_spec__0_spec__1___redArg(v_aesopStx_1604_, v_allGoalsSolved_1606_, v_stats_1605_, v___y_1609_, v___y_1613_, v___y_1614_);
if (lean_obj_tag(v___x_1640_) == 0)
{
lean_object* v_a_1641_; uint8_t v___x_1642_; lean_object* v___x_1643_; 
v_a_1641_ = lean_ctor_get(v___x_1640_, 0);
lean_inc(v_a_1641_);
lean_dec_ref_known(v___x_1640_, 1);
v___x_1642_ = 4;
v___x_1643_ = lean_io_prim_handle_mk(v_file_1637_, v___x_1642_);
lean_dec_ref(v_file_1637_);
if (lean_obj_tag(v___x_1643_) == 0)
{
lean_object* v_a_1644_; uint8_t v___x_1645_; lean_object* v___x_1646_; 
v_a_1644_ = lean_ctor_get(v___x_1643_, 0);
lean_inc(v_a_1644_);
lean_dec_ref_known(v___x_1643_, 1);
v___x_1645_ = 1;
v___x_1646_ = lean_io_prim_handle_lock(v_a_1644_, v___x_1645_);
if (lean_obj_tag(v___x_1646_) == 0)
{
lean_object* v___x_1647_; lean_object* v___x_1648_; lean_object* v_r_1649_; 
lean_dec_ref_known(v___x_1646_, 1);
v___x_1647_ = lp_aesop_Aesop_instToJsonStatsFileRecord_toJson(v_a_1641_);
v___x_1648_ = l_Lean_Json_compress(v___x_1647_);
v_r_1649_ = l_IO_FS_Handle_putStrLn(v_a_1644_, v___x_1648_);
if (lean_obj_tag(v_r_1649_) == 0)
{
lean_object* v_a_1650_; lean_object* v___x_1651_; 
v_a_1650_ = lean_ctor_get(v_r_1649_, 0);
lean_inc(v_a_1650_);
lean_dec_ref_known(v_r_1649_, 1);
v___x_1651_ = lean_io_prim_handle_unlock(v_a_1644_);
lean_dec(v_a_1644_);
if (lean_obj_tag(v___x_1651_) == 0)
{
lean_object* v___x_1653_; uint8_t v_isShared_1654_; uint8_t v_isSharedCheck_1658_; 
v_isSharedCheck_1658_ = !lean_is_exclusive(v___x_1651_);
if (v_isSharedCheck_1658_ == 0)
{
lean_object* v_unused_1659_; 
v_unused_1659_ = lean_ctor_get(v___x_1651_, 0);
lean_dec(v_unused_1659_);
v___x_1653_ = v___x_1651_;
v_isShared_1654_ = v_isSharedCheck_1658_;
goto v_resetjp_1652_;
}
else
{
lean_dec(v___x_1651_);
v___x_1653_ = lean_box(0);
v_isShared_1654_ = v_isSharedCheck_1658_;
goto v_resetjp_1652_;
}
v_resetjp_1652_:
{
lean_object* v___x_1656_; 
if (v_isShared_1654_ == 0)
{
lean_ctor_set(v___x_1653_, 0, v_a_1650_);
v___x_1656_ = v___x_1653_;
goto v_reusejp_1655_;
}
else
{
lean_object* v_reuseFailAlloc_1657_; 
v_reuseFailAlloc_1657_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1657_, 0, v_a_1650_);
v___x_1656_ = v_reuseFailAlloc_1657_;
goto v_reusejp_1655_;
}
v_reusejp_1655_:
{
return v___x_1656_;
}
}
}
else
{
lean_dec(v_a_1650_);
v___y_1626_ = v___x_1651_;
goto v___jp_1625_;
}
}
else
{
lean_object* v_a_1660_; lean_object* v___x_1661_; 
v_a_1660_ = lean_ctor_get(v_r_1649_, 0);
lean_inc(v_a_1660_);
lean_dec_ref_known(v_r_1649_, 1);
v___x_1661_ = lean_io_prim_handle_unlock(v_a_1644_);
lean_dec(v_a_1644_);
if (lean_obj_tag(v___x_1661_) == 0)
{
lean_dec_ref_known(v___x_1661_, 1);
v_a_1619_ = v_a_1660_;
goto v___jp_1618_;
}
else
{
lean_dec(v_a_1660_);
v___y_1626_ = v___x_1661_;
goto v___jp_1625_;
}
}
}
else
{
lean_dec(v_a_1644_);
lean_dec(v_a_1641_);
v___y_1626_ = v___x_1646_;
goto v___jp_1625_;
}
}
else
{
lean_object* v_a_1662_; 
lean_dec(v_a_1641_);
v_a_1662_ = lean_ctor_get(v___x_1643_, 0);
lean_inc(v_a_1662_);
lean_dec_ref_known(v___x_1643_, 1);
v_a_1619_ = v_a_1662_;
goto v___jp_1618_;
}
}
else
{
lean_object* v_a_1663_; lean_object* v___x_1665_; uint8_t v_isShared_1666_; uint8_t v_isSharedCheck_1670_; 
lean_dec_ref(v_file_1637_);
v_a_1663_ = lean_ctor_get(v___x_1640_, 0);
v_isSharedCheck_1670_ = !lean_is_exclusive(v___x_1640_);
if (v_isSharedCheck_1670_ == 0)
{
v___x_1665_ = v___x_1640_;
v_isShared_1666_ = v_isSharedCheck_1670_;
goto v_resetjp_1664_;
}
else
{
lean_inc(v_a_1663_);
lean_dec(v___x_1640_);
v___x_1665_ = lean_box(0);
v_isShared_1666_ = v_isSharedCheck_1670_;
goto v_resetjp_1664_;
}
v_resetjp_1664_:
{
lean_object* v___x_1668_; 
if (v_isShared_1666_ == 0)
{
v___x_1668_ = v___x_1665_;
goto v_reusejp_1667_;
}
else
{
lean_object* v_reuseFailAlloc_1669_; 
v_reuseFailAlloc_1669_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1669_, 0, v_a_1663_);
v___x_1668_ = v_reuseFailAlloc_1669_;
goto v_reusejp_1667_;
}
v_reusejp_1667_:
{
return v___x_1668_;
}
}
}
}
else
{
lean_object* v___x_1671_; lean_object* v___x_1672_; 
lean_dec_ref(v_file_1637_);
lean_dec_ref(v_stats_1605_);
lean_dec(v_aesopStx_1604_);
v___x_1671_ = lean_box(0);
v___x_1672_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1672_, 0, v___x_1671_);
return v___x_1672_;
}
v___jp_1618_:
{
lean_object* v___x_1620_; lean_object* v___x_1621_; lean_object* v___x_1622_; lean_object* v___x_1623_; lean_object* v___x_1624_; 
v___x_1620_ = lean_io_error_to_string(v_a_1619_);
v___x_1621_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_1621_, 0, v___x_1620_);
v___x_1622_ = l_Lean_MessageData_ofFormat(v___x_1621_);
lean_inc(v_ref_1617_);
v___x_1623_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1623_, 0, v_ref_1617_);
lean_ctor_set(v___x_1623_, 1, v___x_1622_);
v___x_1624_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1624_, 0, v___x_1623_);
return v___x_1624_;
}
v___jp_1625_:
{
if (lean_obj_tag(v___y_1626_) == 0)
{
lean_object* v_a_1627_; lean_object* v___x_1629_; uint8_t v_isShared_1630_; uint8_t v_isSharedCheck_1634_; 
v_a_1627_ = lean_ctor_get(v___y_1626_, 0);
v_isSharedCheck_1634_ = !lean_is_exclusive(v___y_1626_);
if (v_isSharedCheck_1634_ == 0)
{
v___x_1629_ = v___y_1626_;
v_isShared_1630_ = v_isSharedCheck_1634_;
goto v_resetjp_1628_;
}
else
{
lean_inc(v_a_1627_);
lean_dec(v___y_1626_);
v___x_1629_ = lean_box(0);
v_isShared_1630_ = v_isSharedCheck_1634_;
goto v_resetjp_1628_;
}
v_resetjp_1628_:
{
lean_object* v___x_1632_; 
if (v_isShared_1630_ == 0)
{
v___x_1632_ = v___x_1629_;
goto v_reusejp_1631_;
}
else
{
lean_object* v_reuseFailAlloc_1633_; 
v_reuseFailAlloc_1633_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1633_, 0, v_a_1627_);
v___x_1632_ = v_reuseFailAlloc_1633_;
goto v_reusejp_1631_;
}
v_reusejp_1631_:
{
return v___x_1632_;
}
}
}
else
{
lean_object* v_a_1635_; 
v_a_1635_ = lean_ctor_get(v___y_1626_, 0);
lean_inc(v_a_1635_);
lean_dec_ref_known(v___y_1626_, 1);
v_a_1619_ = v_a_1635_;
goto v___jp_1618_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_appendStatsToStatsFileIfEnabled___at___00Aesop_Frontend_evalSaturateCore_spec__0___boxed(lean_object* v_aesopStx_1673_, lean_object* v_stats_1674_, lean_object* v_allGoalsSolved_1675_, lean_object* v___y_1676_, lean_object* v___y_1677_, lean_object* v___y_1678_, lean_object* v___y_1679_, lean_object* v___y_1680_, lean_object* v___y_1681_, lean_object* v___y_1682_, lean_object* v___y_1683_, lean_object* v___y_1684_){
_start:
{
uint8_t v_allGoalsSolved_boxed_1685_; lean_object* v_res_1686_; 
v_allGoalsSolved_boxed_1685_ = lean_unbox(v_allGoalsSolved_1675_);
v_res_1686_ = lp_aesop_Aesop_appendStatsToStatsFileIfEnabled___at___00Aesop_Frontend_evalSaturateCore_spec__0(v_aesopStx_1673_, v_stats_1674_, v_allGoalsSolved_boxed_1685_, v___y_1676_, v___y_1677_, v___y_1678_, v___y_1679_, v___y_1680_, v___y_1681_, v___y_1682_, v___y_1683_);
lean_dec(v___y_1683_);
lean_dec_ref(v___y_1682_);
lean_dec(v___y_1681_);
lean_dec_ref(v___y_1680_);
lean_dec(v___y_1679_);
lean_dec_ref(v___y_1678_);
lean_dec(v___y_1677_);
lean_dec_ref(v___y_1676_);
return v_res_1686_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_evalSaturateCore(lean_object* v_stx_1687_, lean_object* v_depth_x3f_1688_, lean_object* v_rules_x3f_1689_, lean_object* v_rs_x3f_1690_, uint8_t v_traceScript_1691_, lean_object* v_a_1692_, lean_object* v_a_1693_, lean_object* v_a_1694_, lean_object* v_a_1695_, lean_object* v_a_1696_, lean_object* v_a_1697_, lean_object* v_a_1698_, lean_object* v_a_1699_){
_start:
{
lean_object* v___y_1702_; 
if (lean_obj_tag(v_depth_x3f_1688_) == 0)
{
lean_object* v___x_1747_; 
v___x_1747_ = lean_box(0);
v___y_1702_ = v___x_1747_;
goto v___jp_1701_;
}
else
{
lean_object* v_val_1748_; lean_object* v___x_1750_; uint8_t v_isShared_1751_; uint8_t v_isSharedCheck_1756_; 
v_val_1748_ = lean_ctor_get(v_depth_x3f_1688_, 0);
v_isSharedCheck_1756_ = !lean_is_exclusive(v_depth_x3f_1688_);
if (v_isSharedCheck_1756_ == 0)
{
v___x_1750_ = v_depth_x3f_1688_;
v_isShared_1751_ = v_isSharedCheck_1756_;
goto v_resetjp_1749_;
}
else
{
lean_inc(v_val_1748_);
lean_dec(v_depth_x3f_1688_);
v___x_1750_ = lean_box(0);
v_isShared_1751_ = v_isSharedCheck_1756_;
goto v_resetjp_1749_;
}
v_resetjp_1749_:
{
lean_object* v___x_1752_; lean_object* v___x_1754_; 
v___x_1752_ = l_Lean_TSyntax_getNat(v_val_1748_);
lean_dec(v_val_1748_);
if (v_isShared_1751_ == 0)
{
lean_ctor_set(v___x_1750_, 0, v___x_1752_);
v___x_1754_ = v___x_1750_;
goto v_reusejp_1753_;
}
else
{
lean_object* v_reuseFailAlloc_1755_; 
v_reuseFailAlloc_1755_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1755_, 0, v___x_1752_);
v___x_1754_ = v_reuseFailAlloc_1755_;
goto v_reusejp_1753_;
}
v_reusejp_1753_:
{
v___y_1702_ = v___x_1754_;
goto v___jp_1701_;
}
}
}
v___jp_1701_:
{
lean_object* v___x_1703_; 
v___x_1703_ = lp_aesop_Aesop_Frontend_mkForwardOptions(v___y_1702_, v_traceScript_1691_, v_a_1698_, v_a_1699_);
if (lean_obj_tag(v___x_1703_) == 0)
{
lean_object* v_a_1704_; lean_object* v___x_1705_; 
v_a_1704_ = lean_ctor_get(v___x_1703_, 0);
lean_inc(v_a_1704_);
lean_dec_ref_known(v___x_1703_, 1);
v___x_1705_ = l_Lean_Elab_Tactic_getMainGoal___redArg(v_a_1693_, v_a_1696_, v_a_1697_, v_a_1698_, v_a_1699_);
if (lean_obj_tag(v___x_1705_) == 0)
{
lean_object* v_a_1706_; lean_object* v___x_1707_; lean_object* v___x_1708_; 
v_a_1706_ = lean_ctor_get(v___x_1705_, 0);
lean_inc(v_a_1706_);
lean_dec_ref_known(v___x_1705_, 1);
lean_inc(v_a_1704_);
v___x_1707_ = lean_alloc_closure((void*)(lp_aesop_Aesop_Frontend_elabForwardRuleSet___boxed), 11, 3);
lean_closure_set(v___x_1707_, 0, v_rs_x3f_1690_);
lean_closure_set(v___x_1707_, 1, v_rules_x3f_1689_);
lean_closure_set(v___x_1707_, 2, v_a_1704_);
v___x_1708_ = lp_aesop_Aesop_ElabM_runForwardElab___redArg(v_a_1706_, v___x_1707_, v_a_1694_, v_a_1695_, v_a_1696_, v_a_1697_, v_a_1698_, v_a_1699_);
if (lean_obj_tag(v___x_1708_) == 0)
{
lean_object* v_a_1709_; lean_object* v___f_1710_; lean_object* v___x_1711_; 
v_a_1709_ = lean_ctor_get(v___x_1708_, 0);
lean_inc(v_a_1709_);
lean_dec_ref_known(v___x_1708_, 1);
v___f_1710_ = lean_alloc_closure((void*)(lp_aesop_Aesop_Frontend_evalSaturateCore___lam__0___boxed), 11, 2);
lean_closure_set(v___f_1710_, 0, v_a_1709_);
lean_closure_set(v___f_1710_, 1, v_a_1704_);
v___x_1711_ = l_Lean_Elab_Tactic_withMainContext___redArg(v___f_1710_, v_a_1692_, v_a_1693_, v_a_1694_, v_a_1695_, v_a_1696_, v_a_1697_, v_a_1698_, v_a_1699_);
if (lean_obj_tag(v___x_1711_) == 0)
{
lean_object* v_a_1712_; uint8_t v___x_1713_; lean_object* v___x_1714_; 
v_a_1712_ = lean_ctor_get(v___x_1711_, 0);
lean_inc(v_a_1712_);
lean_dec_ref_known(v___x_1711_, 1);
v___x_1713_ = 0;
v___x_1714_ = lp_aesop_Aesop_appendStatsToStatsFileIfEnabled___at___00Aesop_Frontend_evalSaturateCore_spec__0(v_stx_1687_, v_a_1712_, v___x_1713_, v_a_1692_, v_a_1693_, v_a_1694_, v_a_1695_, v_a_1696_, v_a_1697_, v_a_1698_, v_a_1699_);
return v___x_1714_;
}
else
{
lean_object* v_a_1715_; lean_object* v___x_1717_; uint8_t v_isShared_1718_; uint8_t v_isSharedCheck_1722_; 
lean_dec(v_stx_1687_);
v_a_1715_ = lean_ctor_get(v___x_1711_, 0);
v_isSharedCheck_1722_ = !lean_is_exclusive(v___x_1711_);
if (v_isSharedCheck_1722_ == 0)
{
v___x_1717_ = v___x_1711_;
v_isShared_1718_ = v_isSharedCheck_1722_;
goto v_resetjp_1716_;
}
else
{
lean_inc(v_a_1715_);
lean_dec(v___x_1711_);
v___x_1717_ = lean_box(0);
v_isShared_1718_ = v_isSharedCheck_1722_;
goto v_resetjp_1716_;
}
v_resetjp_1716_:
{
lean_object* v___x_1720_; 
if (v_isShared_1718_ == 0)
{
v___x_1720_ = v___x_1717_;
goto v_reusejp_1719_;
}
else
{
lean_object* v_reuseFailAlloc_1721_; 
v_reuseFailAlloc_1721_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1721_, 0, v_a_1715_);
v___x_1720_ = v_reuseFailAlloc_1721_;
goto v_reusejp_1719_;
}
v_reusejp_1719_:
{
return v___x_1720_;
}
}
}
}
else
{
lean_object* v_a_1723_; lean_object* v___x_1725_; uint8_t v_isShared_1726_; uint8_t v_isSharedCheck_1730_; 
lean_dec(v_a_1704_);
lean_dec(v_stx_1687_);
v_a_1723_ = lean_ctor_get(v___x_1708_, 0);
v_isSharedCheck_1730_ = !lean_is_exclusive(v___x_1708_);
if (v_isSharedCheck_1730_ == 0)
{
v___x_1725_ = v___x_1708_;
v_isShared_1726_ = v_isSharedCheck_1730_;
goto v_resetjp_1724_;
}
else
{
lean_inc(v_a_1723_);
lean_dec(v___x_1708_);
v___x_1725_ = lean_box(0);
v_isShared_1726_ = v_isSharedCheck_1730_;
goto v_resetjp_1724_;
}
v_resetjp_1724_:
{
lean_object* v___x_1728_; 
if (v_isShared_1726_ == 0)
{
v___x_1728_ = v___x_1725_;
goto v_reusejp_1727_;
}
else
{
lean_object* v_reuseFailAlloc_1729_; 
v_reuseFailAlloc_1729_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1729_, 0, v_a_1723_);
v___x_1728_ = v_reuseFailAlloc_1729_;
goto v_reusejp_1727_;
}
v_reusejp_1727_:
{
return v___x_1728_;
}
}
}
}
else
{
lean_object* v_a_1731_; lean_object* v___x_1733_; uint8_t v_isShared_1734_; uint8_t v_isSharedCheck_1738_; 
lean_dec(v_a_1704_);
lean_dec(v_rs_x3f_1690_);
lean_dec(v_rules_x3f_1689_);
lean_dec(v_stx_1687_);
v_a_1731_ = lean_ctor_get(v___x_1705_, 0);
v_isSharedCheck_1738_ = !lean_is_exclusive(v___x_1705_);
if (v_isSharedCheck_1738_ == 0)
{
v___x_1733_ = v___x_1705_;
v_isShared_1734_ = v_isSharedCheck_1738_;
goto v_resetjp_1732_;
}
else
{
lean_inc(v_a_1731_);
lean_dec(v___x_1705_);
v___x_1733_ = lean_box(0);
v_isShared_1734_ = v_isSharedCheck_1738_;
goto v_resetjp_1732_;
}
v_resetjp_1732_:
{
lean_object* v___x_1736_; 
if (v_isShared_1734_ == 0)
{
v___x_1736_ = v___x_1733_;
goto v_reusejp_1735_;
}
else
{
lean_object* v_reuseFailAlloc_1737_; 
v_reuseFailAlloc_1737_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1737_, 0, v_a_1731_);
v___x_1736_ = v_reuseFailAlloc_1737_;
goto v_reusejp_1735_;
}
v_reusejp_1735_:
{
return v___x_1736_;
}
}
}
}
else
{
lean_object* v_a_1739_; lean_object* v___x_1741_; uint8_t v_isShared_1742_; uint8_t v_isSharedCheck_1746_; 
lean_dec(v_rs_x3f_1690_);
lean_dec(v_rules_x3f_1689_);
lean_dec(v_stx_1687_);
v_a_1739_ = lean_ctor_get(v___x_1703_, 0);
v_isSharedCheck_1746_ = !lean_is_exclusive(v___x_1703_);
if (v_isSharedCheck_1746_ == 0)
{
v___x_1741_ = v___x_1703_;
v_isShared_1742_ = v_isSharedCheck_1746_;
goto v_resetjp_1740_;
}
else
{
lean_inc(v_a_1739_);
lean_dec(v___x_1703_);
v___x_1741_ = lean_box(0);
v_isShared_1742_ = v_isSharedCheck_1746_;
goto v_resetjp_1740_;
}
v_resetjp_1740_:
{
lean_object* v___x_1744_; 
if (v_isShared_1742_ == 0)
{
v___x_1744_ = v___x_1741_;
goto v_reusejp_1743_;
}
else
{
lean_object* v_reuseFailAlloc_1745_; 
v_reuseFailAlloc_1745_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1745_, 0, v_a_1739_);
v___x_1744_ = v_reuseFailAlloc_1745_;
goto v_reusejp_1743_;
}
v_reusejp_1743_:
{
return v___x_1744_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_evalSaturateCore___boxed(lean_object* v_stx_1757_, lean_object* v_depth_x3f_1758_, lean_object* v_rules_x3f_1759_, lean_object* v_rs_x3f_1760_, lean_object* v_traceScript_1761_, lean_object* v_a_1762_, lean_object* v_a_1763_, lean_object* v_a_1764_, lean_object* v_a_1765_, lean_object* v_a_1766_, lean_object* v_a_1767_, lean_object* v_a_1768_, lean_object* v_a_1769_, lean_object* v_a_1770_){
_start:
{
uint8_t v_traceScript_boxed_1771_; lean_object* v_res_1772_; 
v_traceScript_boxed_1771_ = lean_unbox(v_traceScript_1761_);
v_res_1772_ = lp_aesop_Aesop_Frontend_evalSaturateCore(v_stx_1757_, v_depth_x3f_1758_, v_rules_x3f_1759_, v_rs_x3f_1760_, v_traceScript_boxed_1771_, v_a_1762_, v_a_1763_, v_a_1764_, v_a_1765_, v_a_1766_, v_a_1767_, v_a_1768_, v_a_1769_);
lean_dec(v_a_1769_);
lean_dec_ref(v_a_1768_);
lean_dec(v_a_1767_);
lean_dec_ref(v_a_1766_);
lean_dec(v_a_1765_);
lean_dec_ref(v_a_1764_);
lean_dec(v_a_1763_);
lean_dec_ref(v_a_1762_);
return v_res_1772_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_StatsFileRecord_ofStats___at___00Aesop_appendStatsToStatsFileIfEnabled___at___00Aesop_Frontend_evalSaturateCore_spec__0_spec__1(lean_object* v_aesopStx_1773_, uint8_t v_goalSolved_1774_, lean_object* v_stats_1775_, lean_object* v___y_1776_, lean_object* v___y_1777_, lean_object* v___y_1778_, lean_object* v___y_1779_, lean_object* v___y_1780_, lean_object* v___y_1781_, lean_object* v___y_1782_, lean_object* v___y_1783_){
_start:
{
lean_object* v___x_1785_; 
v___x_1785_ = lp_aesop_Aesop_StatsFileRecord_ofStats___at___00Aesop_appendStatsToStatsFileIfEnabled___at___00Aesop_Frontend_evalSaturateCore_spec__0_spec__1___redArg(v_aesopStx_1773_, v_goalSolved_1774_, v_stats_1775_, v___y_1778_, v___y_1782_, v___y_1783_);
return v___x_1785_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_StatsFileRecord_ofStats___at___00Aesop_appendStatsToStatsFileIfEnabled___at___00Aesop_Frontend_evalSaturateCore_spec__0_spec__1___boxed(lean_object* v_aesopStx_1786_, lean_object* v_goalSolved_1787_, lean_object* v_stats_1788_, lean_object* v___y_1789_, lean_object* v___y_1790_, lean_object* v___y_1791_, lean_object* v___y_1792_, lean_object* v___y_1793_, lean_object* v___y_1794_, lean_object* v___y_1795_, lean_object* v___y_1796_, lean_object* v___y_1797_){
_start:
{
uint8_t v_goalSolved_boxed_1798_; lean_object* v_res_1799_; 
v_goalSolved_boxed_1798_ = lean_unbox(v_goalSolved_1787_);
v_res_1799_ = lp_aesop_Aesop_StatsFileRecord_ofStats___at___00Aesop_appendStatsToStatsFileIfEnabled___at___00Aesop_Frontend_evalSaturateCore_spec__0_spec__1(v_aesopStx_1786_, v_goalSolved_boxed_1798_, v_stats_1788_, v___y_1789_, v___y_1790_, v___y_1791_, v___y_1792_, v___y_1793_, v___y_1794_, v___y_1795_, v___y_1796_);
lean_dec(v___y_1796_);
lean_dec_ref(v___y_1795_);
lean_dec(v___y_1794_);
lean_dec_ref(v___y_1793_);
lean_dec(v___y_1792_);
lean_dec_ref(v___y_1791_);
lean_dec(v___y_1790_);
lean_dec_ref(v___y_1789_);
return v_res_1799_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Elab_throwUnsupportedSyntax___at___00Aesop_Frontend_evalSaturate_spec__0___redArg(){
_start:
{
lean_object* v___x_1885_; lean_object* v___x_1886_; 
v___x_1885_ = lean_obj_once(&lp_aesop_Lean_Elab_throwUnsupportedSyntax___at___00Aesop_Frontend_elabAdditionalForwardRules_spec__0___redArg___closed__0, &lp_aesop_Lean_Elab_throwUnsupportedSyntax___at___00Aesop_Frontend_elabAdditionalForwardRules_spec__0___redArg___closed__0_once, _init_lp_aesop_Lean_Elab_throwUnsupportedSyntax___at___00Aesop_Frontend_elabAdditionalForwardRules_spec__0___redArg___closed__0);
v___x_1886_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1886_, 0, v___x_1885_);
return v___x_1886_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Elab_throwUnsupportedSyntax___at___00Aesop_Frontend_evalSaturate_spec__0___redArg___boxed(lean_object* v___y_1887_){
_start:
{
lean_object* v_res_1888_; 
v_res_1888_ = lp_aesop_Lean_Elab_throwUnsupportedSyntax___at___00Aesop_Frontend_evalSaturate_spec__0___redArg();
return v_res_1888_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Elab_throwUnsupportedSyntax___at___00Aesop_Frontend_evalSaturate_spec__0(lean_object* v_00_u03b1_1889_, lean_object* v___y_1890_, lean_object* v___y_1891_, lean_object* v___y_1892_, lean_object* v___y_1893_, lean_object* v___y_1894_, lean_object* v___y_1895_, lean_object* v___y_1896_, lean_object* v___y_1897_){
_start:
{
lean_object* v___x_1899_; 
v___x_1899_ = lp_aesop_Lean_Elab_throwUnsupportedSyntax___at___00Aesop_Frontend_evalSaturate_spec__0___redArg();
return v___x_1899_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Elab_throwUnsupportedSyntax___at___00Aesop_Frontend_evalSaturate_spec__0___boxed(lean_object* v_00_u03b1_1900_, lean_object* v___y_1901_, lean_object* v___y_1902_, lean_object* v___y_1903_, lean_object* v___y_1904_, lean_object* v___y_1905_, lean_object* v___y_1906_, lean_object* v___y_1907_, lean_object* v___y_1908_, lean_object* v___y_1909_){
_start:
{
lean_object* v_res_1910_; 
v_res_1910_ = lp_aesop_Lean_Elab_throwUnsupportedSyntax___at___00Aesop_Frontend_evalSaturate_spec__0(v_00_u03b1_1900_, v___y_1901_, v___y_1902_, v___y_1903_, v___y_1904_, v___y_1905_, v___y_1906_, v___y_1907_, v___y_1908_);
lean_dec(v___y_1908_);
lean_dec_ref(v___y_1907_);
lean_dec(v___y_1906_);
lean_dec_ref(v___y_1905_);
lean_dec(v___y_1904_);
lean_dec_ref(v___y_1903_);
lean_dec(v___y_1902_);
lean_dec_ref(v___y_1901_);
return v_res_1910_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_evalSaturate(lean_object* v_x_1911_, lean_object* v_a_1912_, lean_object* v_a_1913_, lean_object* v_a_1914_, lean_object* v_a_1915_, lean_object* v_a_1916_, lean_object* v_a_1917_, lean_object* v_a_1918_, lean_object* v_a_1919_){
_start:
{
lean_object* v___y_1922_; lean_object* v___y_1923_; lean_object* v_usingRuleSets_x3f_1924_; lean_object* v___y_1925_; lean_object* v___y_1926_; lean_object* v___y_1927_; lean_object* v___y_1928_; lean_object* v___y_1929_; lean_object* v___y_1930_; lean_object* v___y_1931_; lean_object* v___y_1932_; lean_object* v___x_1935_; uint8_t v___x_1936_; 
v___x_1935_ = ((lean_object*)(lp_aesop_Aesop_Frontend_saturate___closed__1));
lean_inc(v_x_1911_);
v___x_1936_ = l_Lean_Syntax_isOfKind(v_x_1911_, v___x_1935_);
if (v___x_1936_ == 0)
{
lean_object* v___x_1937_; 
lean_dec(v_x_1911_);
v___x_1937_ = lp_aesop_Lean_Elab_throwUnsupportedSyntax___at___00Aesop_Frontend_evalSaturate_spec__0___redArg();
return v___x_1937_;
}
else
{
lean_object* v___x_1938_; lean_object* v___x_1939_; lean_object* v___y_1941_; lean_object* v_rules_x3f_1942_; lean_object* v___y_1943_; lean_object* v___y_1944_; lean_object* v___y_1945_; lean_object* v___y_1946_; lean_object* v___y_1947_; lean_object* v___y_1948_; lean_object* v___y_1949_; lean_object* v___y_1950_; lean_object* v_depth_x3f_1960_; lean_object* v___y_1961_; lean_object* v___y_1962_; lean_object* v___y_1963_; lean_object* v___y_1964_; lean_object* v___y_1965_; lean_object* v___y_1966_; lean_object* v___y_1967_; lean_object* v___y_1968_; lean_object* v___x_1977_; uint8_t v___x_1978_; 
v___x_1938_ = lean_unsigned_to_nat(0u);
v___x_1939_ = lean_unsigned_to_nat(1u);
v___x_1977_ = l_Lean_Syntax_getArg(v_x_1911_, v___x_1939_);
v___x_1978_ = l_Lean_Syntax_isNone(v___x_1977_);
if (v___x_1978_ == 0)
{
uint8_t v___x_1979_; 
lean_inc(v___x_1977_);
v___x_1979_ = l_Lean_Syntax_matchesNull(v___x_1977_, v___x_1939_);
if (v___x_1979_ == 0)
{
lean_object* v___x_1980_; 
lean_dec(v___x_1977_);
lean_dec(v_x_1911_);
v___x_1980_ = lp_aesop_Lean_Elab_throwUnsupportedSyntax___at___00Aesop_Frontend_evalSaturate_spec__0___redArg();
return v___x_1980_;
}
else
{
lean_object* v_depth_x3f_1981_; lean_object* v___x_1982_; 
v_depth_x3f_1981_ = l_Lean_Syntax_getArg(v___x_1977_, v___x_1938_);
lean_dec(v___x_1977_);
v___x_1982_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1982_, 0, v_depth_x3f_1981_);
v_depth_x3f_1960_ = v___x_1982_;
v___y_1961_ = v_a_1912_;
v___y_1962_ = v_a_1913_;
v___y_1963_ = v_a_1914_;
v___y_1964_ = v_a_1915_;
v___y_1965_ = v_a_1916_;
v___y_1966_ = v_a_1917_;
v___y_1967_ = v_a_1918_;
v___y_1968_ = v_a_1919_;
goto v___jp_1959_;
}
}
else
{
lean_object* v___x_1983_; 
lean_dec(v___x_1977_);
v___x_1983_ = lean_box(0);
v_depth_x3f_1960_ = v___x_1983_;
v___y_1961_ = v_a_1912_;
v___y_1962_ = v_a_1913_;
v___y_1963_ = v_a_1914_;
v___y_1964_ = v_a_1915_;
v___y_1965_ = v_a_1916_;
v___y_1966_ = v_a_1917_;
v___y_1967_ = v_a_1918_;
v___y_1968_ = v_a_1919_;
goto v___jp_1959_;
}
v___jp_1940_:
{
lean_object* v___x_1951_; lean_object* v___x_1952_; uint8_t v___x_1953_; 
v___x_1951_ = lean_unsigned_to_nat(3u);
v___x_1952_ = l_Lean_Syntax_getArg(v_x_1911_, v___x_1951_);
v___x_1953_ = l_Lean_Syntax_isNone(v___x_1952_);
if (v___x_1953_ == 0)
{
uint8_t v___x_1954_; 
lean_inc(v___x_1952_);
v___x_1954_ = l_Lean_Syntax_matchesNull(v___x_1952_, v___x_1939_);
if (v___x_1954_ == 0)
{
lean_object* v___x_1955_; 
lean_dec(v___x_1952_);
lean_dec(v_rules_x3f_1942_);
lean_dec(v___y_1941_);
lean_dec(v_x_1911_);
v___x_1955_ = lp_aesop_Lean_Elab_throwUnsupportedSyntax___at___00Aesop_Frontend_evalSaturate_spec__0___redArg();
return v___x_1955_;
}
else
{
lean_object* v_usingRuleSets_x3f_1956_; lean_object* v___x_1957_; 
v_usingRuleSets_x3f_1956_ = l_Lean_Syntax_getArg(v___x_1952_, v___x_1938_);
lean_dec(v___x_1952_);
v___x_1957_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1957_, 0, v_usingRuleSets_x3f_1956_);
v___y_1922_ = v___y_1941_;
v___y_1923_ = v_rules_x3f_1942_;
v_usingRuleSets_x3f_1924_ = v___x_1957_;
v___y_1925_ = v___y_1943_;
v___y_1926_ = v___y_1944_;
v___y_1927_ = v___y_1945_;
v___y_1928_ = v___y_1946_;
v___y_1929_ = v___y_1947_;
v___y_1930_ = v___y_1948_;
v___y_1931_ = v___y_1949_;
v___y_1932_ = v___y_1950_;
goto v___jp_1921_;
}
}
else
{
lean_object* v___x_1958_; 
lean_dec(v___x_1952_);
v___x_1958_ = lean_box(0);
v___y_1922_ = v___y_1941_;
v___y_1923_ = v_rules_x3f_1942_;
v_usingRuleSets_x3f_1924_ = v___x_1958_;
v___y_1925_ = v___y_1943_;
v___y_1926_ = v___y_1944_;
v___y_1927_ = v___y_1945_;
v___y_1928_ = v___y_1946_;
v___y_1929_ = v___y_1947_;
v___y_1930_ = v___y_1948_;
v___y_1931_ = v___y_1949_;
v___y_1932_ = v___y_1950_;
goto v___jp_1921_;
}
}
v___jp_1959_:
{
lean_object* v___x_1969_; lean_object* v___x_1970_; uint8_t v___x_1971_; 
v___x_1969_ = lean_unsigned_to_nat(2u);
v___x_1970_ = l_Lean_Syntax_getArg(v_x_1911_, v___x_1969_);
v___x_1971_ = l_Lean_Syntax_isNone(v___x_1970_);
if (v___x_1971_ == 0)
{
uint8_t v___x_1972_; 
lean_inc(v___x_1970_);
v___x_1972_ = l_Lean_Syntax_matchesNull(v___x_1970_, v___x_1939_);
if (v___x_1972_ == 0)
{
lean_object* v___x_1973_; 
lean_dec(v___x_1970_);
lean_dec(v_depth_x3f_1960_);
lean_dec(v_x_1911_);
v___x_1973_ = lp_aesop_Lean_Elab_throwUnsupportedSyntax___at___00Aesop_Frontend_evalSaturate_spec__0___redArg();
return v___x_1973_;
}
else
{
lean_object* v_rules_x3f_1974_; lean_object* v___x_1975_; 
v_rules_x3f_1974_ = l_Lean_Syntax_getArg(v___x_1970_, v___x_1938_);
lean_dec(v___x_1970_);
v___x_1975_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1975_, 0, v_rules_x3f_1974_);
v___y_1941_ = v_depth_x3f_1960_;
v_rules_x3f_1942_ = v___x_1975_;
v___y_1943_ = v___y_1961_;
v___y_1944_ = v___y_1962_;
v___y_1945_ = v___y_1963_;
v___y_1946_ = v___y_1964_;
v___y_1947_ = v___y_1965_;
v___y_1948_ = v___y_1966_;
v___y_1949_ = v___y_1967_;
v___y_1950_ = v___y_1968_;
goto v___jp_1940_;
}
}
else
{
lean_object* v___x_1976_; 
lean_dec(v___x_1970_);
v___x_1976_ = lean_box(0);
v___y_1941_ = v_depth_x3f_1960_;
v_rules_x3f_1942_ = v___x_1976_;
v___y_1943_ = v___y_1961_;
v___y_1944_ = v___y_1962_;
v___y_1945_ = v___y_1963_;
v___y_1946_ = v___y_1964_;
v___y_1947_ = v___y_1965_;
v___y_1948_ = v___y_1966_;
v___y_1949_ = v___y_1967_;
v___y_1950_ = v___y_1968_;
goto v___jp_1940_;
}
}
}
v___jp_1921_:
{
uint8_t v___x_1933_; lean_object* v___x_1934_; 
v___x_1933_ = 0;
v___x_1934_ = lp_aesop_Aesop_Frontend_evalSaturateCore(v_x_1911_, v___y_1922_, v___y_1923_, v_usingRuleSets_x3f_1924_, v___x_1933_, v___y_1925_, v___y_1926_, v___y_1927_, v___y_1928_, v___y_1929_, v___y_1930_, v___y_1931_, v___y_1932_);
return v___x_1934_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_evalSaturate___boxed(lean_object* v_x_1984_, lean_object* v_a_1985_, lean_object* v_a_1986_, lean_object* v_a_1987_, lean_object* v_a_1988_, lean_object* v_a_1989_, lean_object* v_a_1990_, lean_object* v_a_1991_, lean_object* v_a_1992_, lean_object* v_a_1993_){
_start:
{
lean_object* v_res_1994_; 
v_res_1994_ = lp_aesop_Aesop_Frontend_evalSaturate(v_x_1984_, v_a_1985_, v_a_1986_, v_a_1987_, v_a_1988_, v_a_1989_, v_a_1990_, v_a_1991_, v_a_1992_);
lean_dec(v_a_1992_);
lean_dec_ref(v_a_1991_);
lean_dec(v_a_1990_);
lean_dec_ref(v_a_1989_);
lean_dec(v_a_1988_);
lean_dec_ref(v_a_1987_);
lean_dec(v_a_1986_);
lean_dec_ref(v_a_1985_);
return v_res_1994_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_evalSaturate_x3f(lean_object* v_x_1995_, lean_object* v_a_1996_, lean_object* v_a_1997_, lean_object* v_a_1998_, lean_object* v_a_1999_, lean_object* v_a_2000_, lean_object* v_a_2001_, lean_object* v_a_2002_, lean_object* v_a_2003_){
_start:
{
lean_object* v___x_2005_; uint8_t v___x_2006_; 
v___x_2005_ = ((lean_object*)(lp_aesop_Aesop_Frontend_saturate_x3f___closed__1));
lean_inc(v_x_1995_);
v___x_2006_ = l_Lean_Syntax_isOfKind(v_x_1995_, v___x_2005_);
if (v___x_2006_ == 0)
{
lean_object* v___x_2007_; 
lean_dec(v_x_1995_);
v___x_2007_ = lp_aesop_Lean_Elab_throwUnsupportedSyntax___at___00Aesop_Frontend_evalSaturate_spec__0___redArg();
return v___x_2007_;
}
else
{
lean_object* v___x_2008_; lean_object* v___x_2009_; lean_object* v___y_2011_; lean_object* v_rules_x3f_2012_; lean_object* v___y_2013_; lean_object* v___y_2014_; lean_object* v___y_2015_; lean_object* v___y_2016_; lean_object* v___y_2017_; lean_object* v___y_2018_; lean_object* v___y_2019_; lean_object* v___y_2020_; lean_object* v_depth_x3f_2032_; lean_object* v___y_2033_; lean_object* v___y_2034_; lean_object* v___y_2035_; lean_object* v___y_2036_; lean_object* v___y_2037_; lean_object* v___y_2038_; lean_object* v___y_2039_; lean_object* v___y_2040_; lean_object* v___x_2049_; uint8_t v___x_2050_; 
v___x_2008_ = lean_unsigned_to_nat(0u);
v___x_2009_ = lean_unsigned_to_nat(1u);
v___x_2049_ = l_Lean_Syntax_getArg(v_x_1995_, v___x_2009_);
v___x_2050_ = l_Lean_Syntax_isNone(v___x_2049_);
if (v___x_2050_ == 0)
{
uint8_t v___x_2051_; 
lean_inc(v___x_2049_);
v___x_2051_ = l_Lean_Syntax_matchesNull(v___x_2049_, v___x_2009_);
if (v___x_2051_ == 0)
{
lean_object* v___x_2052_; 
lean_dec(v___x_2049_);
lean_dec(v_x_1995_);
v___x_2052_ = lp_aesop_Lean_Elab_throwUnsupportedSyntax___at___00Aesop_Frontend_evalSaturate_spec__0___redArg();
return v___x_2052_;
}
else
{
lean_object* v_depth_x3f_2053_; lean_object* v___x_2054_; 
v_depth_x3f_2053_ = l_Lean_Syntax_getArg(v___x_2049_, v___x_2008_);
lean_dec(v___x_2049_);
v___x_2054_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2054_, 0, v_depth_x3f_2053_);
v_depth_x3f_2032_ = v___x_2054_;
v___y_2033_ = v_a_1996_;
v___y_2034_ = v_a_1997_;
v___y_2035_ = v_a_1998_;
v___y_2036_ = v_a_1999_;
v___y_2037_ = v_a_2000_;
v___y_2038_ = v_a_2001_;
v___y_2039_ = v_a_2002_;
v___y_2040_ = v_a_2003_;
goto v___jp_2031_;
}
}
else
{
lean_object* v___x_2055_; 
lean_dec(v___x_2049_);
v___x_2055_ = lean_box(0);
v_depth_x3f_2032_ = v___x_2055_;
v___y_2033_ = v_a_1996_;
v___y_2034_ = v_a_1997_;
v___y_2035_ = v_a_1998_;
v___y_2036_ = v_a_1999_;
v___y_2037_ = v_a_2000_;
v___y_2038_ = v_a_2001_;
v___y_2039_ = v_a_2002_;
v___y_2040_ = v_a_2003_;
goto v___jp_2031_;
}
v___jp_2010_:
{
lean_object* v___x_2021_; lean_object* v___x_2022_; uint8_t v___x_2023_; 
v___x_2021_ = lean_unsigned_to_nat(3u);
v___x_2022_ = l_Lean_Syntax_getArg(v_x_1995_, v___x_2021_);
v___x_2023_ = l_Lean_Syntax_isNone(v___x_2022_);
if (v___x_2023_ == 0)
{
uint8_t v___x_2024_; 
lean_inc(v___x_2022_);
v___x_2024_ = l_Lean_Syntax_matchesNull(v___x_2022_, v___x_2009_);
if (v___x_2024_ == 0)
{
lean_object* v___x_2025_; 
lean_dec(v___x_2022_);
lean_dec(v_rules_x3f_2012_);
lean_dec(v___y_2011_);
lean_dec(v_x_1995_);
v___x_2025_ = lp_aesop_Lean_Elab_throwUnsupportedSyntax___at___00Aesop_Frontend_evalSaturate_spec__0___redArg();
return v___x_2025_;
}
else
{
lean_object* v_usingRuleSets_x3f_2026_; lean_object* v___x_2027_; lean_object* v___x_2028_; 
v_usingRuleSets_x3f_2026_ = l_Lean_Syntax_getArg(v___x_2022_, v___x_2008_);
lean_dec(v___x_2022_);
v___x_2027_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2027_, 0, v_usingRuleSets_x3f_2026_);
v___x_2028_ = lp_aesop_Aesop_Frontend_evalSaturateCore(v_x_1995_, v___y_2011_, v_rules_x3f_2012_, v___x_2027_, v___x_2006_, v___y_2013_, v___y_2014_, v___y_2015_, v___y_2016_, v___y_2017_, v___y_2018_, v___y_2019_, v___y_2020_);
return v___x_2028_;
}
}
else
{
lean_object* v___x_2029_; lean_object* v___x_2030_; 
lean_dec(v___x_2022_);
v___x_2029_ = lean_box(0);
v___x_2030_ = lp_aesop_Aesop_Frontend_evalSaturateCore(v_x_1995_, v___y_2011_, v_rules_x3f_2012_, v___x_2029_, v___x_2006_, v___y_2013_, v___y_2014_, v___y_2015_, v___y_2016_, v___y_2017_, v___y_2018_, v___y_2019_, v___y_2020_);
return v___x_2030_;
}
}
v___jp_2031_:
{
lean_object* v___x_2041_; lean_object* v___x_2042_; uint8_t v___x_2043_; 
v___x_2041_ = lean_unsigned_to_nat(2u);
v___x_2042_ = l_Lean_Syntax_getArg(v_x_1995_, v___x_2041_);
v___x_2043_ = l_Lean_Syntax_isNone(v___x_2042_);
if (v___x_2043_ == 0)
{
uint8_t v___x_2044_; 
lean_inc(v___x_2042_);
v___x_2044_ = l_Lean_Syntax_matchesNull(v___x_2042_, v___x_2009_);
if (v___x_2044_ == 0)
{
lean_object* v___x_2045_; 
lean_dec(v___x_2042_);
lean_dec(v_depth_x3f_2032_);
lean_dec(v_x_1995_);
v___x_2045_ = lp_aesop_Lean_Elab_throwUnsupportedSyntax___at___00Aesop_Frontend_evalSaturate_spec__0___redArg();
return v___x_2045_;
}
else
{
lean_object* v_rules_x3f_2046_; lean_object* v___x_2047_; 
v_rules_x3f_2046_ = l_Lean_Syntax_getArg(v___x_2042_, v___x_2008_);
lean_dec(v___x_2042_);
v___x_2047_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2047_, 0, v_rules_x3f_2046_);
v___y_2011_ = v_depth_x3f_2032_;
v_rules_x3f_2012_ = v___x_2047_;
v___y_2013_ = v___y_2033_;
v___y_2014_ = v___y_2034_;
v___y_2015_ = v___y_2035_;
v___y_2016_ = v___y_2036_;
v___y_2017_ = v___y_2037_;
v___y_2018_ = v___y_2038_;
v___y_2019_ = v___y_2039_;
v___y_2020_ = v___y_2040_;
goto v___jp_2010_;
}
}
else
{
lean_object* v___x_2048_; 
lean_dec(v___x_2042_);
v___x_2048_ = lean_box(0);
v___y_2011_ = v_depth_x3f_2032_;
v_rules_x3f_2012_ = v___x_2048_;
v___y_2013_ = v___y_2033_;
v___y_2014_ = v___y_2034_;
v___y_2015_ = v___y_2035_;
v___y_2016_ = v___y_2036_;
v___y_2017_ = v___y_2037_;
v___y_2018_ = v___y_2038_;
v___y_2019_ = v___y_2039_;
v___y_2020_ = v___y_2040_;
goto v___jp_2010_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend_evalSaturate_x3f___boxed(lean_object* v_x_2056_, lean_object* v_a_2057_, lean_object* v_a_2058_, lean_object* v_a_2059_, lean_object* v_a_2060_, lean_object* v_a_2061_, lean_object* v_a_2062_, lean_object* v_a_2063_, lean_object* v_a_2064_, lean_object* v_a_2065_){
_start:
{
lean_object* v_res_2066_; 
v_res_2066_ = lp_aesop_Aesop_Frontend_evalSaturate_x3f(v_x_2056_, v_a_2057_, v_a_2058_, v_a_2059_, v_a_2060_, v_a_2061_, v_a_2062_, v_a_2063_, v_a_2064_);
lean_dec(v_a_2064_);
lean_dec_ref(v_a_2063_);
lean_dec(v_a_2062_);
lean_dec_ref(v_a_2061_);
lean_dec(v_a_2060_);
lean_dec_ref(v_a_2059_);
lean_dec(v_a_2058_);
lean_dec_ref(v_a_2057_);
return v_res_2066_;
}
}
static lean_object* _init_lp_aesop_Aesop_Frontend___aux__Aesop__Frontend__Saturate______macroRules__Aesop__Frontend__tacticForward__________1___closed__4(void){
_start:
{
lean_object* v___x_2095_; 
v___x_2095_ = l_Array_mkArray0(lean_box(0));
return v___x_2095_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend___aux__Aesop__Frontend__Saturate______macroRules__Aesop__Frontend__tacticForward__________1(lean_object* v_x_2096_, lean_object* v_a_2097_, lean_object* v_a_2098_){
_start:
{
lean_object* v___y_2100_; lean_object* v___y_2101_; lean_object* v___y_2102_; lean_object* v___y_2103_; lean_object* v___y_2104_; lean_object* v___y_2105_; lean_object* v___y_2106_; lean_object* v___y_2107_; lean_object* v___y_2113_; lean_object* v___y_2114_; lean_object* v___y_2115_; lean_object* v___y_2116_; lean_object* v___y_2117_; lean_object* v___y_2118_; lean_object* v___y_2119_; lean_object* v___y_2120_; lean_object* v___y_2127_; lean_object* v___y_2128_; lean_object* v___x_2145_; uint8_t v___x_2146_; 
v___x_2145_ = ((lean_object*)(lp_aesop_Aesop_Frontend_tacticForward_________00__closed__1));
lean_inc(v_x_2096_);
v___x_2146_ = l_Lean_Syntax_isOfKind(v_x_2096_, v___x_2145_);
if (v___x_2146_ == 0)
{
lean_object* v___x_2147_; lean_object* v___x_2148_; 
lean_dec(v_x_2096_);
v___x_2147_ = lean_box(1);
v___x_2148_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2148_, 0, v___x_2147_);
lean_ctor_set(v___x_2148_, 1, v_a_2098_);
return v___x_2148_;
}
else
{
lean_object* v___x_2149_; lean_object* v___x_2150_; lean_object* v___y_2152_; lean_object* v___x_2163_; lean_object* v___x_2164_; lean_object* v___x_2165_; 
v___x_2149_ = lean_unsigned_to_nat(1u);
v___x_2150_ = l_Lean_Syntax_getArg(v_x_2096_, v___x_2149_);
v___x_2163_ = lean_unsigned_to_nat(2u);
v___x_2164_ = l_Lean_Syntax_getArg(v_x_2096_, v___x_2163_);
lean_dec(v_x_2096_);
v___x_2165_ = l_Lean_Syntax_getOptional_x3f(v___x_2164_);
lean_dec(v___x_2164_);
if (lean_obj_tag(v___x_2165_) == 0)
{
lean_object* v___x_2166_; 
v___x_2166_ = lean_box(0);
v___y_2152_ = v___x_2166_;
goto v___jp_2151_;
}
else
{
lean_object* v_val_2167_; lean_object* v___x_2169_; uint8_t v_isShared_2170_; uint8_t v_isSharedCheck_2174_; 
v_val_2167_ = lean_ctor_get(v___x_2165_, 0);
v_isSharedCheck_2174_ = !lean_is_exclusive(v___x_2165_);
if (v_isSharedCheck_2174_ == 0)
{
v___x_2169_ = v___x_2165_;
v_isShared_2170_ = v_isSharedCheck_2174_;
goto v_resetjp_2168_;
}
else
{
lean_inc(v_val_2167_);
lean_dec(v___x_2165_);
v___x_2169_ = lean_box(0);
v_isShared_2170_ = v_isSharedCheck_2174_;
goto v_resetjp_2168_;
}
v_resetjp_2168_:
{
lean_object* v___x_2172_; 
if (v_isShared_2170_ == 0)
{
v___x_2172_ = v___x_2169_;
goto v_reusejp_2171_;
}
else
{
lean_object* v_reuseFailAlloc_2173_; 
v_reuseFailAlloc_2173_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2173_, 0, v_val_2167_);
v___x_2172_ = v_reuseFailAlloc_2173_;
goto v_reusejp_2171_;
}
v_reusejp_2171_:
{
v___y_2152_ = v___x_2172_;
goto v___jp_2151_;
}
}
}
v___jp_2151_:
{
lean_object* v___x_2153_; 
v___x_2153_ = l_Lean_Syntax_getOptional_x3f(v___x_2150_);
lean_dec(v___x_2150_);
if (lean_obj_tag(v___x_2153_) == 0)
{
lean_object* v___x_2154_; 
v___x_2154_ = lean_box(0);
v___y_2127_ = v___y_2152_;
v___y_2128_ = v___x_2154_;
goto v___jp_2126_;
}
else
{
lean_object* v_val_2155_; lean_object* v___x_2157_; uint8_t v_isShared_2158_; uint8_t v_isSharedCheck_2162_; 
v_val_2155_ = lean_ctor_get(v___x_2153_, 0);
v_isSharedCheck_2162_ = !lean_is_exclusive(v___x_2153_);
if (v_isSharedCheck_2162_ == 0)
{
v___x_2157_ = v___x_2153_;
v_isShared_2158_ = v_isSharedCheck_2162_;
goto v_resetjp_2156_;
}
else
{
lean_inc(v_val_2155_);
lean_dec(v___x_2153_);
v___x_2157_ = lean_box(0);
v_isShared_2158_ = v_isSharedCheck_2162_;
goto v_resetjp_2156_;
}
v_resetjp_2156_:
{
lean_object* v___x_2160_; 
if (v_isShared_2158_ == 0)
{
v___x_2160_ = v___x_2157_;
goto v_reusejp_2159_;
}
else
{
lean_object* v_reuseFailAlloc_2161_; 
v_reuseFailAlloc_2161_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2161_, 0, v_val_2155_);
v___x_2160_ = v_reuseFailAlloc_2161_;
goto v_reusejp_2159_;
}
v_reusejp_2159_:
{
v___y_2127_ = v___y_2152_;
v___y_2128_ = v___x_2160_;
goto v___jp_2126_;
}
}
}
}
}
v___jp_2099_:
{
lean_object* v___x_2108_; lean_object* v___x_2109_; lean_object* v___x_2110_; lean_object* v___x_2111_; 
lean_inc_ref(v___y_2103_);
v___x_2108_ = l_Array_append___redArg(v___y_2103_, v___y_2107_);
lean_dec_ref(v___y_2107_);
lean_inc(v___y_2105_);
v___x_2109_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_2109_, 0, v___y_2105_);
lean_ctor_set(v___x_2109_, 1, v___y_2104_);
lean_ctor_set(v___x_2109_, 2, v___x_2108_);
lean_inc(v___y_2100_);
v___x_2110_ = l_Lean_Syntax_node4(v___y_2105_, v___y_2100_, v___y_2106_, v___y_2102_, v___y_2101_, v___x_2109_);
v___x_2111_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2111_, 0, v___x_2110_);
lean_ctor_set(v___x_2111_, 1, v_a_2098_);
return v___x_2111_;
}
v___jp_2112_:
{
lean_object* v___x_2121_; lean_object* v___x_2122_; 
lean_inc_ref(v___y_2116_);
v___x_2121_ = l_Array_append___redArg(v___y_2116_, v___y_2120_);
lean_dec_ref(v___y_2120_);
lean_inc(v___y_2117_);
lean_inc(v___y_2118_);
v___x_2122_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_2122_, 0, v___y_2118_);
lean_ctor_set(v___x_2122_, 1, v___y_2117_);
lean_ctor_set(v___x_2122_, 2, v___x_2121_);
if (lean_obj_tag(v___y_2113_) == 1)
{
lean_object* v_val_2123_; lean_object* v___x_2124_; 
v_val_2123_ = lean_ctor_get(v___y_2113_, 0);
lean_inc(v_val_2123_);
lean_dec_ref_known(v___y_2113_, 1);
v___x_2124_ = l_Array_mkArray1___redArg(v_val_2123_);
v___y_2100_ = v___y_2114_;
v___y_2101_ = v___x_2122_;
v___y_2102_ = v___y_2115_;
v___y_2103_ = v___y_2116_;
v___y_2104_ = v___y_2117_;
v___y_2105_ = v___y_2118_;
v___y_2106_ = v___y_2119_;
v___y_2107_ = v___x_2124_;
goto v___jp_2099_;
}
else
{
lean_object* v___x_2125_; 
lean_dec(v___y_2113_);
v___x_2125_ = ((lean_object*)(lp_aesop_Aesop_Frontend___aux__Aesop__Frontend__Saturate______macroRules__Aesop__Frontend__tacticForward__________1___closed__0));
v___y_2100_ = v___y_2114_;
v___y_2101_ = v___x_2122_;
v___y_2102_ = v___y_2115_;
v___y_2103_ = v___y_2116_;
v___y_2104_ = v___y_2117_;
v___y_2105_ = v___y_2118_;
v___y_2106_ = v___y_2119_;
v___y_2107_ = v___x_2125_;
goto v___jp_2099_;
}
}
v___jp_2126_:
{
lean_object* v_ref_2129_; uint8_t v___x_2130_; lean_object* v___x_2131_; lean_object* v___x_2132_; lean_object* v___x_2133_; lean_object* v___x_2134_; lean_object* v___x_2135_; lean_object* v___x_2136_; lean_object* v___x_2137_; lean_object* v___x_2138_; lean_object* v___x_2139_; lean_object* v___x_2140_; lean_object* v___x_2141_; 
v_ref_2129_ = lean_ctor_get(v_a_2097_, 5);
v___x_2130_ = 0;
v___x_2131_ = l_Lean_SourceInfo_fromRef(v_ref_2129_, v___x_2130_);
v___x_2132_ = ((lean_object*)(lp_aesop_Aesop_Frontend_saturate___closed__0));
v___x_2133_ = ((lean_object*)(lp_aesop_Aesop_Frontend_saturate___closed__1));
lean_inc_n(v___x_2131_, 4);
v___x_2134_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2134_, 0, v___x_2131_);
lean_ctor_set(v___x_2134_, 1, v___x_2132_);
v___x_2135_ = ((lean_object*)(lp_aesop_Aesop_Frontend___aux__Aesop__Frontend__Saturate______macroRules__Aesop__Frontend__tacticForward__________1___closed__2));
v___x_2136_ = ((lean_object*)(lp_aesop_Aesop_Frontend_saturate___closed__9));
v___x_2137_ = ((lean_object*)(lp_aesop_Aesop_Frontend___aux__Aesop__Frontend__Saturate______macroRules__Aesop__Frontend__tacticForward__________1___closed__3));
v___x_2138_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2138_, 0, v___x_2131_);
lean_ctor_set(v___x_2138_, 1, v___x_2137_);
v___x_2139_ = l_Lean_Syntax_node1(v___x_2131_, v___x_2136_, v___x_2138_);
v___x_2140_ = l_Lean_Syntax_node1(v___x_2131_, v___x_2135_, v___x_2139_);
v___x_2141_ = lean_obj_once(&lp_aesop_Aesop_Frontend___aux__Aesop__Frontend__Saturate______macroRules__Aesop__Frontend__tacticForward__________1___closed__4, &lp_aesop_Aesop_Frontend___aux__Aesop__Frontend__Saturate______macroRules__Aesop__Frontend__tacticForward__________1___closed__4_once, _init_lp_aesop_Aesop_Frontend___aux__Aesop__Frontend__Saturate______macroRules__Aesop__Frontend__tacticForward__________1___closed__4);
if (lean_obj_tag(v___y_2128_) == 1)
{
lean_object* v_val_2142_; lean_object* v___x_2143_; 
v_val_2142_ = lean_ctor_get(v___y_2128_, 0);
lean_inc(v_val_2142_);
lean_dec_ref_known(v___y_2128_, 1);
v___x_2143_ = l_Array_mkArray1___redArg(v_val_2142_);
v___y_2113_ = v___y_2127_;
v___y_2114_ = v___x_2133_;
v___y_2115_ = v___x_2140_;
v___y_2116_ = v___x_2141_;
v___y_2117_ = v___x_2135_;
v___y_2118_ = v___x_2131_;
v___y_2119_ = v___x_2134_;
v___y_2120_ = v___x_2143_;
goto v___jp_2112_;
}
else
{
lean_object* v___x_2144_; 
lean_dec(v___y_2128_);
v___x_2144_ = ((lean_object*)(lp_aesop_Aesop_Frontend___aux__Aesop__Frontend__Saturate______macroRules__Aesop__Frontend__tacticForward__________1___closed__0));
v___y_2113_ = v___y_2127_;
v___y_2114_ = v___x_2133_;
v___y_2115_ = v___x_2140_;
v___y_2116_ = v___x_2141_;
v___y_2117_ = v___x_2135_;
v___y_2118_ = v___x_2131_;
v___y_2119_ = v___x_2134_;
v___y_2120_ = v___x_2144_;
goto v___jp_2112_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend___aux__Aesop__Frontend__Saturate______macroRules__Aesop__Frontend__tacticForward__________1___boxed(lean_object* v_x_2175_, lean_object* v_a_2176_, lean_object* v_a_2177_){
_start:
{
lean_object* v_res_2178_; 
v_res_2178_ = lp_aesop_Aesop_Frontend___aux__Aesop__Frontend__Saturate______macroRules__Aesop__Frontend__tacticForward__________1(v_x_2175_, v_a_2176_, v_a_2177_);
lean_dec_ref(v_a_2176_);
return v_res_2178_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend___aux__Aesop__Frontend__Saturate______macroRules__Aesop__Frontend__tacticForward_x3f__________1(lean_object* v_x_2201_, lean_object* v_a_2202_, lean_object* v_a_2203_){
_start:
{
lean_object* v___y_2205_; lean_object* v___y_2206_; lean_object* v___y_2207_; lean_object* v___y_2208_; lean_object* v___y_2209_; lean_object* v___y_2210_; lean_object* v___y_2211_; lean_object* v___y_2212_; lean_object* v___y_2218_; lean_object* v___y_2219_; lean_object* v___y_2220_; lean_object* v___y_2221_; lean_object* v___y_2222_; lean_object* v___y_2223_; lean_object* v___y_2224_; lean_object* v___y_2225_; lean_object* v___y_2232_; lean_object* v___y_2233_; lean_object* v___x_2250_; uint8_t v___x_2251_; 
v___x_2250_ = ((lean_object*)(lp_aesop_Aesop_Frontend_tacticForward_x3f_________00__closed__1));
lean_inc(v_x_2201_);
v___x_2251_ = l_Lean_Syntax_isOfKind(v_x_2201_, v___x_2250_);
if (v___x_2251_ == 0)
{
lean_object* v___x_2252_; lean_object* v___x_2253_; 
lean_dec(v_x_2201_);
v___x_2252_ = lean_box(1);
v___x_2253_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2253_, 0, v___x_2252_);
lean_ctor_set(v___x_2253_, 1, v_a_2203_);
return v___x_2253_;
}
else
{
lean_object* v___x_2254_; lean_object* v___x_2255_; lean_object* v___y_2257_; lean_object* v___x_2268_; lean_object* v___x_2269_; lean_object* v___x_2270_; 
v___x_2254_ = lean_unsigned_to_nat(1u);
v___x_2255_ = l_Lean_Syntax_getArg(v_x_2201_, v___x_2254_);
v___x_2268_ = lean_unsigned_to_nat(2u);
v___x_2269_ = l_Lean_Syntax_getArg(v_x_2201_, v___x_2268_);
lean_dec(v_x_2201_);
v___x_2270_ = l_Lean_Syntax_getOptional_x3f(v___x_2269_);
lean_dec(v___x_2269_);
if (lean_obj_tag(v___x_2270_) == 0)
{
lean_object* v___x_2271_; 
v___x_2271_ = lean_box(0);
v___y_2257_ = v___x_2271_;
goto v___jp_2256_;
}
else
{
lean_object* v_val_2272_; lean_object* v___x_2274_; uint8_t v_isShared_2275_; uint8_t v_isSharedCheck_2279_; 
v_val_2272_ = lean_ctor_get(v___x_2270_, 0);
v_isSharedCheck_2279_ = !lean_is_exclusive(v___x_2270_);
if (v_isSharedCheck_2279_ == 0)
{
v___x_2274_ = v___x_2270_;
v_isShared_2275_ = v_isSharedCheck_2279_;
goto v_resetjp_2273_;
}
else
{
lean_inc(v_val_2272_);
lean_dec(v___x_2270_);
v___x_2274_ = lean_box(0);
v_isShared_2275_ = v_isSharedCheck_2279_;
goto v_resetjp_2273_;
}
v_resetjp_2273_:
{
lean_object* v___x_2277_; 
if (v_isShared_2275_ == 0)
{
v___x_2277_ = v___x_2274_;
goto v_reusejp_2276_;
}
else
{
lean_object* v_reuseFailAlloc_2278_; 
v_reuseFailAlloc_2278_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2278_, 0, v_val_2272_);
v___x_2277_ = v_reuseFailAlloc_2278_;
goto v_reusejp_2276_;
}
v_reusejp_2276_:
{
v___y_2257_ = v___x_2277_;
goto v___jp_2256_;
}
}
}
v___jp_2256_:
{
lean_object* v___x_2258_; 
v___x_2258_ = l_Lean_Syntax_getOptional_x3f(v___x_2255_);
lean_dec(v___x_2255_);
if (lean_obj_tag(v___x_2258_) == 0)
{
lean_object* v___x_2259_; 
v___x_2259_ = lean_box(0);
v___y_2232_ = v___y_2257_;
v___y_2233_ = v___x_2259_;
goto v___jp_2231_;
}
else
{
lean_object* v_val_2260_; lean_object* v___x_2262_; uint8_t v_isShared_2263_; uint8_t v_isSharedCheck_2267_; 
v_val_2260_ = lean_ctor_get(v___x_2258_, 0);
v_isSharedCheck_2267_ = !lean_is_exclusive(v___x_2258_);
if (v_isSharedCheck_2267_ == 0)
{
v___x_2262_ = v___x_2258_;
v_isShared_2263_ = v_isSharedCheck_2267_;
goto v_resetjp_2261_;
}
else
{
lean_inc(v_val_2260_);
lean_dec(v___x_2258_);
v___x_2262_ = lean_box(0);
v_isShared_2263_ = v_isSharedCheck_2267_;
goto v_resetjp_2261_;
}
v_resetjp_2261_:
{
lean_object* v___x_2265_; 
if (v_isShared_2263_ == 0)
{
v___x_2265_ = v___x_2262_;
goto v_reusejp_2264_;
}
else
{
lean_object* v_reuseFailAlloc_2266_; 
v_reuseFailAlloc_2266_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2266_, 0, v_val_2260_);
v___x_2265_ = v_reuseFailAlloc_2266_;
goto v_reusejp_2264_;
}
v_reusejp_2264_:
{
v___y_2232_ = v___y_2257_;
v___y_2233_ = v___x_2265_;
goto v___jp_2231_;
}
}
}
}
}
v___jp_2204_:
{
lean_object* v___x_2213_; lean_object* v___x_2214_; lean_object* v___x_2215_; lean_object* v___x_2216_; 
lean_inc_ref(v___y_2210_);
v___x_2213_ = l_Array_append___redArg(v___y_2210_, v___y_2212_);
lean_dec_ref(v___y_2212_);
lean_inc(v___y_2209_);
v___x_2214_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_2214_, 0, v___y_2209_);
lean_ctor_set(v___x_2214_, 1, v___y_2208_);
lean_ctor_set(v___x_2214_, 2, v___x_2213_);
lean_inc(v___y_2205_);
v___x_2215_ = l_Lean_Syntax_node4(v___y_2209_, v___y_2205_, v___y_2207_, v___y_2211_, v___y_2206_, v___x_2214_);
v___x_2216_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2216_, 0, v___x_2215_);
lean_ctor_set(v___x_2216_, 1, v_a_2203_);
return v___x_2216_;
}
v___jp_2217_:
{
lean_object* v___x_2226_; lean_object* v___x_2227_; 
lean_inc_ref(v___y_2222_);
v___x_2226_ = l_Array_append___redArg(v___y_2222_, v___y_2225_);
lean_dec_ref(v___y_2225_);
lean_inc(v___y_2220_);
lean_inc(v___y_2221_);
v___x_2227_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_2227_, 0, v___y_2221_);
lean_ctor_set(v___x_2227_, 1, v___y_2220_);
lean_ctor_set(v___x_2227_, 2, v___x_2226_);
if (lean_obj_tag(v___y_2223_) == 1)
{
lean_object* v_val_2228_; lean_object* v___x_2229_; 
v_val_2228_ = lean_ctor_get(v___y_2223_, 0);
lean_inc(v_val_2228_);
lean_dec_ref_known(v___y_2223_, 1);
v___x_2229_ = l_Array_mkArray1___redArg(v_val_2228_);
v___y_2205_ = v___y_2218_;
v___y_2206_ = v___x_2227_;
v___y_2207_ = v___y_2219_;
v___y_2208_ = v___y_2220_;
v___y_2209_ = v___y_2221_;
v___y_2210_ = v___y_2222_;
v___y_2211_ = v___y_2224_;
v___y_2212_ = v___x_2229_;
goto v___jp_2204_;
}
else
{
lean_object* v___x_2230_; 
lean_dec(v___y_2223_);
v___x_2230_ = ((lean_object*)(lp_aesop_Aesop_Frontend___aux__Aesop__Frontend__Saturate______macroRules__Aesop__Frontend__tacticForward__________1___closed__0));
v___y_2205_ = v___y_2218_;
v___y_2206_ = v___x_2227_;
v___y_2207_ = v___y_2219_;
v___y_2208_ = v___y_2220_;
v___y_2209_ = v___y_2221_;
v___y_2210_ = v___y_2222_;
v___y_2211_ = v___y_2224_;
v___y_2212_ = v___x_2230_;
goto v___jp_2204_;
}
}
v___jp_2231_:
{
lean_object* v_ref_2234_; uint8_t v___x_2235_; lean_object* v___x_2236_; lean_object* v___x_2237_; lean_object* v___x_2238_; lean_object* v___x_2239_; lean_object* v___x_2240_; lean_object* v___x_2241_; lean_object* v___x_2242_; lean_object* v___x_2243_; lean_object* v___x_2244_; lean_object* v___x_2245_; lean_object* v___x_2246_; 
v_ref_2234_ = lean_ctor_get(v_a_2202_, 5);
v___x_2235_ = 0;
v___x_2236_ = l_Lean_SourceInfo_fromRef(v_ref_2234_, v___x_2235_);
v___x_2237_ = ((lean_object*)(lp_aesop_Aesop_Frontend_saturate_x3f___closed__0));
v___x_2238_ = ((lean_object*)(lp_aesop_Aesop_Frontend_saturate_x3f___closed__1));
lean_inc_n(v___x_2236_, 4);
v___x_2239_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2239_, 0, v___x_2236_);
lean_ctor_set(v___x_2239_, 1, v___x_2237_);
v___x_2240_ = ((lean_object*)(lp_aesop_Aesop_Frontend___aux__Aesop__Frontend__Saturate______macroRules__Aesop__Frontend__tacticForward__________1___closed__2));
v___x_2241_ = ((lean_object*)(lp_aesop_Aesop_Frontend_saturate___closed__9));
v___x_2242_ = ((lean_object*)(lp_aesop_Aesop_Frontend___aux__Aesop__Frontend__Saturate______macroRules__Aesop__Frontend__tacticForward__________1___closed__3));
v___x_2243_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2243_, 0, v___x_2236_);
lean_ctor_set(v___x_2243_, 1, v___x_2242_);
v___x_2244_ = l_Lean_Syntax_node1(v___x_2236_, v___x_2241_, v___x_2243_);
v___x_2245_ = l_Lean_Syntax_node1(v___x_2236_, v___x_2240_, v___x_2244_);
v___x_2246_ = lean_obj_once(&lp_aesop_Aesop_Frontend___aux__Aesop__Frontend__Saturate______macroRules__Aesop__Frontend__tacticForward__________1___closed__4, &lp_aesop_Aesop_Frontend___aux__Aesop__Frontend__Saturate______macroRules__Aesop__Frontend__tacticForward__________1___closed__4_once, _init_lp_aesop_Aesop_Frontend___aux__Aesop__Frontend__Saturate______macroRules__Aesop__Frontend__tacticForward__________1___closed__4);
if (lean_obj_tag(v___y_2233_) == 1)
{
lean_object* v_val_2247_; lean_object* v___x_2248_; 
v_val_2247_ = lean_ctor_get(v___y_2233_, 0);
lean_inc(v_val_2247_);
lean_dec_ref_known(v___y_2233_, 1);
v___x_2248_ = l_Array_mkArray1___redArg(v_val_2247_);
v___y_2218_ = v___x_2238_;
v___y_2219_ = v___x_2239_;
v___y_2220_ = v___x_2240_;
v___y_2221_ = v___x_2236_;
v___y_2222_ = v___x_2246_;
v___y_2223_ = v___y_2232_;
v___y_2224_ = v___x_2245_;
v___y_2225_ = v___x_2248_;
goto v___jp_2217_;
}
else
{
lean_object* v___x_2249_; 
lean_dec(v___y_2233_);
v___x_2249_ = ((lean_object*)(lp_aesop_Aesop_Frontend___aux__Aesop__Frontend__Saturate______macroRules__Aesop__Frontend__tacticForward__________1___closed__0));
v___y_2218_ = v___x_2238_;
v___y_2219_ = v___x_2239_;
v___y_2220_ = v___x_2240_;
v___y_2221_ = v___x_2236_;
v___y_2222_ = v___x_2246_;
v___y_2223_ = v___y_2232_;
v___y_2224_ = v___x_2245_;
v___y_2225_ = v___x_2249_;
goto v___jp_2217_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Frontend___aux__Aesop__Frontend__Saturate______macroRules__Aesop__Frontend__tacticForward_x3f__________1___boxed(lean_object* v_x_2280_, lean_object* v_a_2281_, lean_object* v_a_2282_){
_start:
{
lean_object* v_res_2283_; 
v_res_2283_ = lp_aesop_Aesop_Frontend___aux__Aesop__Frontend__Saturate______macroRules__Aesop__Frontend__tacticForward_x3f__________1(v_x_2280_, v_a_2281_, v_a_2282_);
lean_dec_ref(v_a_2281_);
return v_res_2283_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
void lean_initialize_runtime_module();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_aesop_Aesop_Frontend_Saturate(uint8_t builtin) {
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
lean_object* runtime_initialize_aesop_Aesop_Saturate(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_Frontend_Extension(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_Builder_Forward(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_Stats_File(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_aesop_Aesop_Frontend_Saturate(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Saturate(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Frontend_Extension(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Builder_Forward(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Stats_File(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_aesop_Aesop_Saturate(uint8_t builtin);
lean_object* initialize_aesop_Aesop_Frontend_Extension(uint8_t builtin);
lean_object* initialize_aesop_Aesop_Builder_Forward(uint8_t builtin);
lean_object* initialize_aesop_Aesop_Stats_File(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_aesop_Aesop_Frontend_Saturate(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Saturate(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Frontend_Extension(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Builder_Forward(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Stats_File(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Frontend_Saturate(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_aesop_Aesop_Frontend_Saturate(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_aesop_Aesop_Frontend_Saturate(builtin);
}
#ifdef __cplusplus
}
#endif
