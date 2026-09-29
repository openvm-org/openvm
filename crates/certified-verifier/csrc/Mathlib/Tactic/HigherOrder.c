// Lean compiler output
// Module: Mathlib.Tactic.HigherOrder
// Imports: public import Init public meta import Init public meta import Lean.Meta.Tactic.Apply public meta import Lean.Meta.Tactic.Assumption public meta import Lean.Meta.MatchUtil public meta import Lean.Meta.Tactic.Intro public meta import Lean.Elab.DeclarationRange public import Lean.Meta.Tactic.Simp public import Mathlib.Init
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
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
lean_object* lean_st_ref_get(lean_object*);
lean_object* l_Lean_Elab_getBetterRef(lean_object*, lean_object*);
extern lean_object* l_Lean_Elab_pp_macroStack;
lean_object* l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(lean_object*, lean_object*);
lean_object* l_Lean_MessageData_ofFormat(lean_object*);
lean_object* l_Lean_MessageData_ofSyntax(lean_object*);
lean_object* l_Lean_indentD(lean_object*);
lean_object* l_List_reverse___redArg(lean_object*);
lean_object* l_Lean_mkLevelParam(lean_object*);
size_t lean_uint64_to_usize(uint64_t);
size_t lean_usize_land(size_t, size_t);
lean_object* lean_usize_to_nat(size_t);
lean_object* lean_array_get_borrowed(lean_object*, lean_object*, lean_object*);
uint8_t lean_name_eq(lean_object*, lean_object*);
lean_object* l_Lean_Meta_Origin_key(lean_object*);
size_t lean_usize_shift_right(size_t, size_t);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* lean_array_get_size(lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
lean_object* lean_array_fget_borrowed(lean_object*, lean_object*);
uint64_t lean_uint64_mix_hash(uint64_t, uint64_t);
lean_object* l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_object*, lean_object*);
lean_object* l_Lean_Environment_findConstVal_x3f(lean_object*, lean_object*, uint8_t);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* l_Lean_MessageData_ofConstName(lean_object*, uint8_t);
uint8_t l_Lean_Name_isAnonymous(lean_object*);
lean_object* l_Lean_Environment_setExporting(lean_object*, uint8_t);
uint8_t l_Lean_Environment_contains(lean_object*, lean_object*, uint8_t);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
extern lean_object* l_Lean_Options_empty;
lean_object* l_Lean_Environment_getModuleIdxFor_x3f(lean_object*, lean_object*);
lean_object* l_Lean_MessageData_note(lean_object*);
lean_object* l_Lean_Environment_header(lean_object*);
lean_object* l_Lean_EnvironmentHeader_moduleNames(lean_object*);
lean_object* lean_array_get(lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_isPrivateName(lean_object*);
lean_object* l_Lean_MessageData_ofName(lean_object*);
extern lean_object* l_Lean_unknownIdentifierMessageTag;
lean_object* l_Lean_mkConst(lean_object*, lean_object*);
lean_object* l_Lean_Expr_bindingBody_x21(lean_object*);
lean_object* lean_expr_instantiate1(lean_object*, lean_object*);
uint8_t l_Lean_Expr_isForall(lean_object*);
lean_object* l_Lean_Meta_matchEq_x3f(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_infer_type(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkAppOptM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkAppM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t lean_expr_equal(lean_object*, lean_object*);
uint8_t l_Lean_Expr_occurs(lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_ppExpr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_bindingName_x21(lean_object*);
uint8_t l_Lean_Expr_binderInfo(lean_object*);
lean_object* l_Lean_Expr_bindingDomain_x21(lean_object*);
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_withLocalDeclImp(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkForallFVars(lean_object*, lean_object*, uint8_t, uint8_t, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Expr_hasMVar(lean_object*);
lean_object* l_Lean_instantiateMVarsCore(lean_object*, lean_object*);
lean_object* lean_st_ref_take(lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
extern lean_object* l_Lean_declRangeExt;
lean_object* l_Lean_MapDeclarationExtension_insert___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_Elab_unsupportedSyntaxExceptionId;
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
uint64_t l___private_Lean_Meta_Basic_0__Lean_Meta_Config_toKey(lean_object*);
lean_object* lean_st_mk_ref(lean_object*);
lean_object* l_Lean_Environment_find_x3f(lean_object*, lean_object*, uint8_t);
lean_object* l_Lean_ConstantInfo_levelParams(lean_object*);
lean_object* l_Lean_Expr_const___override(lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkFreshExprMVar(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_mvarId_x21(lean_object*);
lean_object* l_Lean_MVarId_intros(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Term_mkConst(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_getSimpExtension_x3f(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_ScopedEnvExtension_getState___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_addSimpTheorem(lean_object*, lean_object*, uint8_t, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_addDecl(lean_object*, uint8_t, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getRange_x3f(lean_object*, uint8_t);
lean_object* l_Lean_DeclarationRange_ofStringPositions(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Term_addTermInfo_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_Meta_simpExtension;
extern lean_object* l_Lean_Meta_instInhabitedSimpTheorems_default;
lean_object* l_Lean_MVarId_apply(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_intro1Core(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MVarId_assumption(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Term_TermElabM_run___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isNone(lean_object*);
uint8_t l_Lean_Syntax_matchesNull(lean_object*, lean_object*);
lean_object* l_Lean_TSyntax_getId(lean_object*);
lean_object* l_Lean_Name_getPrefix(lean_object*);
lean_object* l_Lean_Name_updatePrefix(lean_object*, lean_object*);
lean_object* lean_name_append_after(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr3(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_registerParametricAttribute___redArg(lean_object*);
static const lean_string_object lp_mathlib_Lean_Parser_Attr_higherOrder___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib_Lean_Parser_Attr_higherOrder___closed__0 = (const lean_object*)&lp_mathlib_Lean_Parser_Attr_higherOrder___closed__0_value;
static const lean_string_object lp_mathlib_Lean_Parser_Attr_higherOrder___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib_Lean_Parser_Attr_higherOrder___closed__1 = (const lean_object*)&lp_mathlib_Lean_Parser_Attr_higherOrder___closed__1_value;
static const lean_string_object lp_mathlib_Lean_Parser_Attr_higherOrder___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Attr"};
static const lean_object* lp_mathlib_Lean_Parser_Attr_higherOrder___closed__2 = (const lean_object*)&lp_mathlib_Lean_Parser_Attr_higherOrder___closed__2_value;
static const lean_string_object lp_mathlib_Lean_Parser_Attr_higherOrder___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "higherOrder"};
static const lean_object* lp_mathlib_Lean_Parser_Attr_higherOrder___closed__3 = (const lean_object*)&lp_mathlib_Lean_Parser_Attr_higherOrder___closed__3_value;
static const lean_ctor_object lp_mathlib_Lean_Parser_Attr_higherOrder___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Parser_Attr_higherOrder___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Lean_Parser_Attr_higherOrder___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Parser_Attr_higherOrder___closed__4_value_aux_0),((lean_object*)&lp_mathlib_Lean_Parser_Attr_higherOrder___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Lean_Parser_Attr_higherOrder___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Parser_Attr_higherOrder___closed__4_value_aux_1),((lean_object*)&lp_mathlib_Lean_Parser_Attr_higherOrder___closed__2_value),LEAN_SCALAR_PTR_LITERAL(7, 175, 252, 195, 22, 42, 161, 63)}};
static const lean_ctor_object lp_mathlib_Lean_Parser_Attr_higherOrder___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Parser_Attr_higherOrder___closed__4_value_aux_2),((lean_object*)&lp_mathlib_Lean_Parser_Attr_higherOrder___closed__3_value),LEAN_SCALAR_PTR_LITERAL(209, 191, 158, 158, 47, 215, 81, 125)}};
static const lean_object* lp_mathlib_Lean_Parser_Attr_higherOrder___closed__4 = (const lean_object*)&lp_mathlib_Lean_Parser_Attr_higherOrder___closed__4_value;
static const lean_string_object lp_mathlib_Lean_Parser_Attr_higherOrder___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_Lean_Parser_Attr_higherOrder___closed__5 = (const lean_object*)&lp_mathlib_Lean_Parser_Attr_higherOrder___closed__5_value;
static const lean_ctor_object lp_mathlib_Lean_Parser_Attr_higherOrder___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Parser_Attr_higherOrder___closed__5_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_Lean_Parser_Attr_higherOrder___closed__6 = (const lean_object*)&lp_mathlib_Lean_Parser_Attr_higherOrder___closed__6_value;
static const lean_string_object lp_mathlib_Lean_Parser_Attr_higherOrder___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "higher_order"};
static const lean_object* lp_mathlib_Lean_Parser_Attr_higherOrder___closed__7 = (const lean_object*)&lp_mathlib_Lean_Parser_Attr_higherOrder___closed__7_value;
static const lean_ctor_object lp_mathlib_Lean_Parser_Attr_higherOrder___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Parser_Attr_higherOrder___closed__7_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Lean_Parser_Attr_higherOrder___closed__8 = (const lean_object*)&lp_mathlib_Lean_Parser_Attr_higherOrder___closed__8_value;
static const lean_string_object lp_mathlib_Lean_Parser_Attr_higherOrder___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "optional"};
static const lean_object* lp_mathlib_Lean_Parser_Attr_higherOrder___closed__9 = (const lean_object*)&lp_mathlib_Lean_Parser_Attr_higherOrder___closed__9_value;
static const lean_ctor_object lp_mathlib_Lean_Parser_Attr_higherOrder___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Parser_Attr_higherOrder___closed__9_value),LEAN_SCALAR_PTR_LITERAL(233, 141, 154, 50, 143, 135, 42, 252)}};
static const lean_object* lp_mathlib_Lean_Parser_Attr_higherOrder___closed__10 = (const lean_object*)&lp_mathlib_Lean_Parser_Attr_higherOrder___closed__10_value;
static const lean_string_object lp_mathlib_Lean_Parser_Attr_higherOrder___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "ppSpace"};
static const lean_object* lp_mathlib_Lean_Parser_Attr_higherOrder___closed__11 = (const lean_object*)&lp_mathlib_Lean_Parser_Attr_higherOrder___closed__11_value;
static const lean_ctor_object lp_mathlib_Lean_Parser_Attr_higherOrder___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Parser_Attr_higherOrder___closed__11_value),LEAN_SCALAR_PTR_LITERAL(207, 47, 58, 43, 30, 240, 125, 246)}};
static const lean_object* lp_mathlib_Lean_Parser_Attr_higherOrder___closed__12 = (const lean_object*)&lp_mathlib_Lean_Parser_Attr_higherOrder___closed__12_value;
static const lean_ctor_object lp_mathlib_Lean_Parser_Attr_higherOrder___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Parser_Attr_higherOrder___closed__12_value)}};
static const lean_object* lp_mathlib_Lean_Parser_Attr_higherOrder___closed__13 = (const lean_object*)&lp_mathlib_Lean_Parser_Attr_higherOrder___closed__13_value;
static const lean_string_object lp_mathlib_Lean_Parser_Attr_higherOrder___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ident"};
static const lean_object* lp_mathlib_Lean_Parser_Attr_higherOrder___closed__14 = (const lean_object*)&lp_mathlib_Lean_Parser_Attr_higherOrder___closed__14_value;
static const lean_ctor_object lp_mathlib_Lean_Parser_Attr_higherOrder___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Parser_Attr_higherOrder___closed__14_value),LEAN_SCALAR_PTR_LITERAL(52, 159, 208, 51, 14, 60, 6, 71)}};
static const lean_object* lp_mathlib_Lean_Parser_Attr_higherOrder___closed__15 = (const lean_object*)&lp_mathlib_Lean_Parser_Attr_higherOrder___closed__15_value;
static const lean_ctor_object lp_mathlib_Lean_Parser_Attr_higherOrder___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Parser_Attr_higherOrder___closed__15_value)}};
static const lean_object* lp_mathlib_Lean_Parser_Attr_higherOrder___closed__16 = (const lean_object*)&lp_mathlib_Lean_Parser_Attr_higherOrder___closed__16_value;
static const lean_ctor_object lp_mathlib_Lean_Parser_Attr_higherOrder___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Parser_Attr_higherOrder___closed__6_value),((lean_object*)&lp_mathlib_Lean_Parser_Attr_higherOrder___closed__13_value),((lean_object*)&lp_mathlib_Lean_Parser_Attr_higherOrder___closed__16_value)}};
static const lean_object* lp_mathlib_Lean_Parser_Attr_higherOrder___closed__17 = (const lean_object*)&lp_mathlib_Lean_Parser_Attr_higherOrder___closed__17_value;
static const lean_ctor_object lp_mathlib_Lean_Parser_Attr_higherOrder___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Parser_Attr_higherOrder___closed__10_value),((lean_object*)&lp_mathlib_Lean_Parser_Attr_higherOrder___closed__17_value)}};
static const lean_object* lp_mathlib_Lean_Parser_Attr_higherOrder___closed__18 = (const lean_object*)&lp_mathlib_Lean_Parser_Attr_higherOrder___closed__18_value;
static const lean_ctor_object lp_mathlib_Lean_Parser_Attr_higherOrder___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Parser_Attr_higherOrder___closed__6_value),((lean_object*)&lp_mathlib_Lean_Parser_Attr_higherOrder___closed__8_value),((lean_object*)&lp_mathlib_Lean_Parser_Attr_higherOrder___closed__18_value)}};
static const lean_object* lp_mathlib_Lean_Parser_Attr_higherOrder___closed__19 = (const lean_object*)&lp_mathlib_Lean_Parser_Attr_higherOrder___closed__19_value;
static const lean_ctor_object lp_mathlib_Lean_Parser_Attr_higherOrder___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Parser_Attr_higherOrder___closed__4_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Parser_Attr_higherOrder___closed__19_value)}};
static const lean_object* lp_mathlib_Lean_Parser_Attr_higherOrder___closed__20 = (const lean_object*)&lp_mathlib_Lean_Parser_Attr_higherOrder___closed__20_value;
LEAN_EXPORT const lean_object* lp_mathlib_Lean_Parser_Attr_higherOrder = (const lean_object*)&lp_mathlib_Lean_Parser_Attr_higherOrder___closed__20_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_mkComp_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_mkComp_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_mkComp_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_mkComp_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_mkComp___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "id"};
static const lean_object* lp_mathlib_Mathlib_Tactic_mkComp___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_mkComp___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_mkComp___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_mkComp___closed__0_value),LEAN_SCALAR_PTR_LITERAL(223, 78, 141, 85, 50, 255, 216, 83)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_mkComp___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_mkComp___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_mkComp___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "Function"};
static const lean_object* lp_mathlib_Mathlib_Tactic_mkComp___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_mkComp___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_mkComp___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "comp"};
static const lean_object* lp_mathlib_Mathlib_Tactic_mkComp___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_mkComp___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_mkComp___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_mkComp___closed__2_value),LEAN_SCALAR_PTR_LITERAL(225, 8, 186, 189, 152, 89, 197, 12)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_mkComp___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_mkComp___closed__4_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_mkComp___closed__3_value),LEAN_SCALAR_PTR_LITERAL(38, 235, 97, 97, 37, 43, 137, 69)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_mkComp___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_mkComp___closed__4_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_mkComp___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 27, .m_capacity = 27, .m_length = 26, .m_data = "mkComp failed occurs check"};
static const lean_object* lp_mathlib_Mathlib_Tactic_mkComp___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_mkComp___closed__5_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_mkComp___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_mkComp___closed__6;
static const lean_string_object lp_mathlib_Mathlib_Tactic_mkComp___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "failed"};
static const lean_object* lp_mathlib_Mathlib_Tactic_mkComp___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_mkComp___closed__7_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_mkComp___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_mkComp___closed__8;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_mkComp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_mkComp___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_mkComp_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_mkComp_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Mathlib_Tactic_mkHigherOrderType_spec__0___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Mathlib_Tactic_mkHigherOrderType_spec__0___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Mathlib_Tactic_mkHigherOrderType_spec__0___redArg(lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Mathlib_Tactic_mkHigherOrderType_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Mathlib_Tactic_mkHigherOrderType_spec__0(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Mathlib_Tactic_mkHigherOrderType_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_mkHigherOrderType___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "not a forall"};
static const lean_object* lp_mathlib_Mathlib_Tactic_mkHigherOrderType___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_mkHigherOrderType___closed__0_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_mkHigherOrderType___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_mkHigherOrderType___closed__1;
static const lean_string_object lp_mathlib_Mathlib_Tactic_mkHigherOrderType___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "not an equality "};
static const lean_object* lp_mathlib_Mathlib_Tactic_mkHigherOrderType___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_mkHigherOrderType___lam__0___closed__0_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_mkHigherOrderType___lam__0___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_mkHigherOrderType___lam__0___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_mkHigherOrderType___lam__0(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_mkHigherOrderType___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_mkHigherOrderType(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_mkHigherOrderType___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_higherOrderGetParam_spec__0___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_higherOrderGetParam_spec__0___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_higherOrderGetParam_spec__0___redArg();
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_higherOrderGetParam_spec__0___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_higherOrderGetParam_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_higherOrderGetParam_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_higherOrderGetParam_spec__3___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_higherOrderGetParam_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_higherOrderGetParam_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_higherOrderGetParam_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Tactic_higherOrderGetParam___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_higherOrderGetParam___lam__0___boxed(lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Mathlib_Tactic_higherOrderGetParam_spec__6_spec__10_spec__12___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Mathlib_Tactic_higherOrderGetParam_spec__6_spec__10_spec__12___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Mathlib_Tactic_higherOrderGetParam_spec__6_spec__10___redArg(lean_object*, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Mathlib_Tactic_higherOrderGetParam_spec__6_spec__10___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_contains___at___00Mathlib_Tactic_higherOrderGetParam_spec__6___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_contains___at___00Mathlib_Tactic_higherOrderGetParam_spec__6___redArg___boxed(lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3_spec__12_spec__17_spec__20___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3_spec__12_spec__17_spec__20___redArg___closed__0;
static lean_once_cell_t lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3_spec__12_spec__17_spec__20___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3_spec__12_spec__17_spec__20___redArg___closed__1;
static lean_once_cell_t lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3_spec__12_spec__17_spec__20___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3_spec__12_spec__17_spec__20___redArg___closed__2;
static lean_once_cell_t lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3_spec__12_spec__17_spec__20___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3_spec__12_spec__17_spec__20___redArg___closed__3;
static lean_once_cell_t lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3_spec__12_spec__17_spec__20___redArg___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3_spec__12_spec__17_spec__20___redArg___closed__4;
static lean_once_cell_t lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3_spec__12_spec__17_spec__20___redArg___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3_spec__12_spec__17_spec__20___redArg___closed__5;
static const lean_string_object lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3_spec__12_spec__17_spec__20___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 24, .m_capacity = 24, .m_length = 23, .m_data = "A private declaration `"};
static const lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3_spec__12_spec__17_spec__20___redArg___closed__6 = (const lean_object*)&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3_spec__12_spec__17_spec__20___redArg___closed__6_value;
static lean_once_cell_t lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3_spec__12_spec__17_spec__20___redArg___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3_spec__12_spec__17_spec__20___redArg___closed__7;
static const lean_string_object lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3_spec__12_spec__17_spec__20___redArg___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 79, .m_capacity = 79, .m_length = 78, .m_data = "` (from the current module) exists but would need to be public to access here."};
static const lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3_spec__12_spec__17_spec__20___redArg___closed__8 = (const lean_object*)&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3_spec__12_spec__17_spec__20___redArg___closed__8_value;
static lean_once_cell_t lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3_spec__12_spec__17_spec__20___redArg___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3_spec__12_spec__17_spec__20___redArg___closed__9;
static const lean_string_object lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3_spec__12_spec__17_spec__20___redArg___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 23, .m_capacity = 23, .m_length = 22, .m_data = "A public declaration `"};
static const lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3_spec__12_spec__17_spec__20___redArg___closed__10 = (const lean_object*)&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3_spec__12_spec__17_spec__20___redArg___closed__10_value;
static lean_once_cell_t lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3_spec__12_spec__17_spec__20___redArg___closed__11_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3_spec__12_spec__17_spec__20___redArg___closed__11;
static const lean_string_object lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3_spec__12_spec__17_spec__20___redArg___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 68, .m_capacity = 68, .m_length = 67, .m_data = "` exists but is imported privately; consider adding `public import "};
static const lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3_spec__12_spec__17_spec__20___redArg___closed__12 = (const lean_object*)&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3_spec__12_spec__17_spec__20___redArg___closed__12_value;
static lean_once_cell_t lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3_spec__12_spec__17_spec__20___redArg___closed__13_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3_spec__12_spec__17_spec__20___redArg___closed__13;
static const lean_string_object lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3_spec__12_spec__17_spec__20___redArg___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "`."};
static const lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3_spec__12_spec__17_spec__20___redArg___closed__14 = (const lean_object*)&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3_spec__12_spec__17_spec__20___redArg___closed__14_value;
static lean_once_cell_t lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3_spec__12_spec__17_spec__20___redArg___closed__15_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3_spec__12_spec__17_spec__20___redArg___closed__15;
static const lean_string_object lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3_spec__12_spec__17_spec__20___redArg___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "` (from `"};
static const lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3_spec__12_spec__17_spec__20___redArg___closed__16 = (const lean_object*)&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3_spec__12_spec__17_spec__20___redArg___closed__16_value;
static lean_once_cell_t lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3_spec__12_spec__17_spec__20___redArg___closed__17_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3_spec__12_spec__17_spec__20___redArg___closed__17;
static const lean_string_object lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3_spec__12_spec__17_spec__20___redArg___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 54, .m_capacity = 54, .m_length = 53, .m_data = "`) exists but would need to be public to access here."};
static const lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3_spec__12_spec__17_spec__20___redArg___closed__18 = (const lean_object*)&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3_spec__12_spec__17_spec__20___redArg___closed__18_value;
static lean_once_cell_t lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3_spec__12_spec__17_spec__20___redArg___closed__19_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3_spec__12_spec__17_spec__20___redArg___closed__19;
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3_spec__12_spec__17_spec__20___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3_spec__12_spec__17_spec__20___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3_spec__12_spec__17(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3_spec__12_spec__17___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_higherOrderGetParam_spec__7_spec__12_spec__15(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_higherOrderGetParam_spec__7_spec__12_spec__15___boxed(lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_higherOrderGetParam_spec__7_spec__12_spec__16___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_higherOrderGetParam_spec__7_spec__12_spec__16___closed__0;
static const lean_string_object lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_higherOrderGetParam_spec__7_spec__12_spec__16___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "while expanding"};
static const lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_higherOrderGetParam_spec__7_spec__12_spec__16___closed__1 = (const lean_object*)&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_higherOrderGetParam_spec__7_spec__12_spec__16___closed__1_value;
static const lean_ctor_object lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_higherOrderGetParam_spec__7_spec__12_spec__16___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_higherOrderGetParam_spec__7_spec__12_spec__16___closed__1_value)}};
static const lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_higherOrderGetParam_spec__7_spec__12_spec__16___closed__2 = (const lean_object*)&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_higherOrderGetParam_spec__7_spec__12_spec__16___closed__2_value;
static lean_once_cell_t lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_higherOrderGetParam_spec__7_spec__12_spec__16___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_higherOrderGetParam_spec__7_spec__12_spec__16___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_higherOrderGetParam_spec__7_spec__12_spec__16(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_higherOrderGetParam_spec__7_spec__12___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 25, .m_capacity = 25, .m_length = 24, .m_data = "with resulting expansion"};
static const lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_higherOrderGetParam_spec__7_spec__12___redArg___closed__0 = (const lean_object*)&lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_higherOrderGetParam_spec__7_spec__12___redArg___closed__0_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_higherOrderGetParam_spec__7_spec__12___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_higherOrderGetParam_spec__7_spec__12___redArg___closed__0_value)}};
static const lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_higherOrderGetParam_spec__7_spec__12___redArg___closed__1 = (const lean_object*)&lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_higherOrderGetParam_spec__7_spec__12___redArg___closed__1_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_higherOrderGetParam_spec__7_spec__12___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_higherOrderGetParam_spec__7_spec__12___redArg___closed__2;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_higherOrderGetParam_spec__7_spec__12___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_higherOrderGetParam_spec__7_spec__12___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_higherOrderGetParam_spec__7___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_higherOrderGetParam_spec__7___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3_spec__12_spec__18___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3_spec__12_spec__18___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3_spec__12___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3_spec__12___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "Unknown constant `"};
static const lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3___redArg___closed__0 = (const lean_object*)&lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3___redArg___closed__0_value;
static lean_once_cell_t lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3___redArg___closed__1;
static const lean_string_object lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "`"};
static const lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3___redArg___closed__2 = (const lean_object*)&lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3___redArg___closed__2_value;
static lean_once_cell_t lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3___redArg___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_higherOrderGetParam_spec__5_spec__8(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_higherOrderGetParam_spec__5_spec__8___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_higherOrderGetParam_spec__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_higherOrderGetParam_spec__5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_higherOrderGetParam_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_addDeclarationRanges___at___00Lean_Elab_addDeclarationRangesFromSyntax___at___00Mathlib_Tactic_higherOrderGetParam_spec__4_spec__6___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addDeclarationRanges___at___00Lean_Elab_addDeclarationRangesFromSyntax___at___00Mathlib_Tactic_higherOrderGetParam_spec__4_spec__6___redArg___closed__0;
static lean_once_cell_t lp_mathlib_Lean_addDeclarationRanges___at___00Lean_Elab_addDeclarationRangesFromSyntax___at___00Mathlib_Tactic_higherOrderGetParam_spec__4_spec__6___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addDeclarationRanges___at___00Lean_Elab_addDeclarationRangesFromSyntax___at___00Mathlib_Tactic_higherOrderGetParam_spec__4_spec__6___redArg___closed__1;
static lean_once_cell_t lp_mathlib_Lean_addDeclarationRanges___at___00Lean_Elab_addDeclarationRangesFromSyntax___at___00Mathlib_Tactic_higherOrderGetParam_spec__4_spec__6___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addDeclarationRanges___at___00Lean_Elab_addDeclarationRangesFromSyntax___at___00Mathlib_Tactic_higherOrderGetParam_spec__4_spec__6___redArg___closed__2;
static lean_once_cell_t lp_mathlib_Lean_addDeclarationRanges___at___00Lean_Elab_addDeclarationRangesFromSyntax___at___00Mathlib_Tactic_higherOrderGetParam_spec__4_spec__6___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addDeclarationRanges___at___00Lean_Elab_addDeclarationRangesFromSyntax___at___00Mathlib_Tactic_higherOrderGetParam_spec__4_spec__6___redArg___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_Lean_addDeclarationRanges___at___00Lean_Elab_addDeclarationRangesFromSyntax___at___00Mathlib_Tactic_higherOrderGetParam_spec__4_spec__6___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addDeclarationRanges___at___00Lean_Elab_addDeclarationRangesFromSyntax___at___00Mathlib_Tactic_higherOrderGetParam_spec__4_spec__6___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_getDeclarationRange_x3f___at___00Lean_Elab_addDeclarationRangesFromSyntax___at___00Mathlib_Tactic_higherOrderGetParam_spec__4_spec__5___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_getDeclarationRange_x3f___at___00Lean_Elab_addDeclarationRangesFromSyntax___at___00Mathlib_Tactic_higherOrderGetParam_spec__4_spec__5___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addDeclarationRangesFromSyntax___at___00Mathlib_Tactic_higherOrderGetParam_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addDeclarationRangesFromSyntax___at___00Mathlib_Tactic_higherOrderGetParam_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forM___at___00Mathlib_Tactic_higherOrderGetParam_spec__8___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forM___at___00Mathlib_Tactic_higherOrderGetParam_spec__8___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_higherOrderGetParam___lam__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "funext"};
static const lean_object* lp_mathlib_Mathlib_Tactic_higherOrderGetParam___lam__1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_higherOrderGetParam___lam__1___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_higherOrderGetParam___lam__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_higherOrderGetParam___lam__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(226, 251, 226, 140, 5, 134, 146, 130)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_higherOrderGetParam___lam__1___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_higherOrderGetParam___lam__1___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_higherOrderGetParam___lam__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "functor_norm"};
static const lean_object* lp_mathlib_Mathlib_Tactic_higherOrderGetParam___lam__1___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_higherOrderGetParam___lam__1___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_higherOrderGetParam___lam__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_higherOrderGetParam___lam__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(231, 168, 30, 17, 22, 99, 221, 74)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_higherOrderGetParam___lam__1___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_higherOrderGetParam___lam__1___closed__3_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_higherOrderGetParam___lam__1(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_higherOrderGetParam___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_Tactic_higherOrderGetParam___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Tactic_higherOrderGetParam___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_higherOrderGetParam___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_higherOrderGetParam___closed__0_value;
static const lean_array_object lp_mathlib_Mathlib_Tactic_higherOrderGetParam___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Mathlib_Tactic_higherOrderGetParam___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_higherOrderGetParam___closed__1_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_higherOrderGetParam___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_higherOrderGetParam___closed__2;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_higherOrderGetParam___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_higherOrderGetParam___closed__3;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_higherOrderGetParam___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_higherOrderGetParam___closed__4;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_higherOrderGetParam___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_higherOrderGetParam___closed__5;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_higherOrderGetParam___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_higherOrderGetParam___closed__6;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_higherOrderGetParam___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_higherOrderGetParam___closed__7;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_higherOrderGetParam___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_higherOrderGetParam___closed__8;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_higherOrderGetParam___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*7 + 0, .m_other = 7, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(1) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(1) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_higherOrderGetParam___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_higherOrderGetParam___closed__9_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_higherOrderGetParam___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "'"};
static const lean_object* lp_mathlib_Mathlib_Tactic_higherOrderGetParam___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_higherOrderGetParam___closed__10_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_higherOrderGetParam(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_higherOrderGetParam___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_getDeclarationRange_x3f___at___00Lean_Elab_addDeclarationRangesFromSyntax___at___00Mathlib_Tactic_higherOrderGetParam_spec__4_spec__5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_getDeclarationRange_x3f___at___00Lean_Elab_addDeclarationRangesFromSyntax___at___00Mathlib_Tactic_higherOrderGetParam_spec__4_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addDeclarationRanges___at___00Lean_Elab_addDeclarationRangesFromSyntax___at___00Mathlib_Tactic_higherOrderGetParam_spec__4_spec__6(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addDeclarationRanges___at___00Lean_Elab_addDeclarationRangesFromSyntax___at___00Mathlib_Tactic_higherOrderGetParam_spec__4_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_contains___at___00Mathlib_Tactic_higherOrderGetParam_spec__6(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_contains___at___00Mathlib_Tactic_higherOrderGetParam_spec__6___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_higherOrderGetParam_spec__7(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_higherOrderGetParam_spec__7___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forM___at___00Mathlib_Tactic_higherOrderGetParam_spec__8(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forM___at___00Mathlib_Tactic_higherOrderGetParam_spec__8___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Mathlib_Tactic_higherOrderGetParam_spec__6_spec__10(lean_object*, lean_object*, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Mathlib_Tactic_higherOrderGetParam_spec__6_spec__10___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_higherOrderGetParam_spec__7_spec__12(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_higherOrderGetParam_spec__7_spec__12___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Mathlib_Tactic_higherOrderGetParam_spec__6_spec__10_spec__12(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Mathlib_Tactic_higherOrderGetParam_spec__6_spec__10_spec__12___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3_spec__12(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3_spec__12___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3_spec__12_spec__17_spec__20(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3_spec__12_spec__17_spec__20___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3_spec__12_spec__18(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3_spec__12_spec__18___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_HigherOrder_0__Mathlib_Tactic_initFn___lam__0_00___x40_Mathlib_Tactic_HigherOrder_2684385311____hygCtx___hyg_2_(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_HigherOrder_0__Mathlib_Tactic_initFn___lam__0_00___x40_Mathlib_Tactic_HigherOrder_2684385311____hygCtx___hyg_2____boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib___private_Mathlib_Tactic_HigherOrder_0__Mathlib_Tactic_initFn___lam__1_00___x40_Mathlib_Tactic_HigherOrder_2684385311____hygCtx___hyg_2_(uint8_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_HigherOrder_0__Mathlib_Tactic_initFn___lam__1_00___x40_Mathlib_Tactic_HigherOrder_2684385311____hygCtx___hyg_2____boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_HigherOrder_0__Mathlib_Tactic_initFn___closed__0_00___x40_Mathlib_Tactic_HigherOrder_2684385311____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_HigherOrder_0__Mathlib_Tactic_initFn___lam__0_00___x40_Mathlib_Tactic_HigherOrder_2684385311____hygCtx___hyg_2____boxed, .m_arity = 5, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_HigherOrder_0__Mathlib_Tactic_initFn___closed__0_00___x40_Mathlib_Tactic_HigherOrder_2684385311____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_HigherOrder_0__Mathlib_Tactic_initFn___closed__0_00___x40_Mathlib_Tactic_HigherOrder_2684385311____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_HigherOrder_0__Mathlib_Tactic_initFn___closed__1_00___x40_Mathlib_Tactic_HigherOrder_2684385311____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_HigherOrder_0__Mathlib_Tactic_initFn___closed__1_00___x40_Mathlib_Tactic_HigherOrder_2684385311____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_HigherOrder_0__Mathlib_Tactic_initFn___closed__1_00___x40_Mathlib_Tactic_HigherOrder_2684385311____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_HigherOrder_0__Mathlib_Tactic_initFn___closed__2_00___x40_Mathlib_Tactic_HigherOrder_2684385311____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_HigherOrder_0__Mathlib_Tactic_initFn___closed__2_00___x40_Mathlib_Tactic_HigherOrder_2684385311____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_HigherOrder_0__Mathlib_Tactic_initFn___closed__2_00___x40_Mathlib_Tactic_HigherOrder_2684385311____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_HigherOrder_0__Mathlib_Tactic_initFn___closed__3_00___x40_Mathlib_Tactic_HigherOrder_2684385311____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "higherOrderAttr"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_HigherOrder_0__Mathlib_Tactic_initFn___closed__3_00___x40_Mathlib_Tactic_HigherOrder_2684385311____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_HigherOrder_0__Mathlib_Tactic_initFn___closed__3_00___x40_Mathlib_Tactic_HigherOrder_2684385311____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_HigherOrder_0__Mathlib_Tactic_initFn___closed__4_00___x40_Mathlib_Tactic_HigherOrder_2684385311____hygCtx___hyg_2__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_HigherOrder_0__Mathlib_Tactic_initFn___closed__1_00___x40_Mathlib_Tactic_HigherOrder_2684385311____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_HigherOrder_0__Mathlib_Tactic_initFn___closed__4_00___x40_Mathlib_Tactic_HigherOrder_2684385311____hygCtx___hyg_2__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_HigherOrder_0__Mathlib_Tactic_initFn___closed__4_00___x40_Mathlib_Tactic_HigherOrder_2684385311____hygCtx___hyg_2__value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_HigherOrder_0__Mathlib_Tactic_initFn___closed__2_00___x40_Mathlib_Tactic_HigherOrder_2684385311____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_HigherOrder_0__Mathlib_Tactic_initFn___closed__4_00___x40_Mathlib_Tactic_HigherOrder_2684385311____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_HigherOrder_0__Mathlib_Tactic_initFn___closed__4_00___x40_Mathlib_Tactic_HigherOrder_2684385311____hygCtx___hyg_2__value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_HigherOrder_0__Mathlib_Tactic_initFn___closed__3_00___x40_Mathlib_Tactic_HigherOrder_2684385311____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(80, 30, 48, 177, 55, 184, 239, 186)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_HigherOrder_0__Mathlib_Tactic_initFn___closed__4_00___x40_Mathlib_Tactic_HigherOrder_2684385311____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_HigherOrder_0__Mathlib_Tactic_initFn___closed__4_00___x40_Mathlib_Tactic_HigherOrder_2684385311____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_HigherOrder_0__Mathlib_Tactic_initFn___closed__5_00___x40_Mathlib_Tactic_HigherOrder_2684385311____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Parser_Attr_higherOrder___closed__3_value),LEAN_SCALAR_PTR_LITERAL(69, 192, 71, 118, 144, 112, 223, 208)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_HigherOrder_0__Mathlib_Tactic_initFn___closed__5_00___x40_Mathlib_Tactic_HigherOrder_2684385311____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_HigherOrder_0__Mathlib_Tactic_initFn___closed__5_00___x40_Mathlib_Tactic_HigherOrder_2684385311____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_HigherOrder_0__Mathlib_Tactic_initFn___closed__6_00___x40_Mathlib_Tactic_HigherOrder_2684385311____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 253, .m_capacity = 253, .m_length = 248, .m_data = "From a lemma of the shape `∀ x, f (g x) = h x` derive an auxiliary lemma of the\nform `f ∘ g = h` for reasoning about higher-order functions.\n\nSyntax: `[higher_order]` or `[higher_order name]`, where the given name is used for the\ngenerated theorem."};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_HigherOrder_0__Mathlib_Tactic_initFn___closed__6_00___x40_Mathlib_Tactic_HigherOrder_2684385311____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_HigherOrder_0__Mathlib_Tactic_initFn___closed__6_00___x40_Mathlib_Tactic_HigherOrder_2684385311____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_HigherOrder_0__Mathlib_Tactic_initFn___closed__7_00___x40_Mathlib_Tactic_HigherOrder_2684385311____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 8, .m_other = 3, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_HigherOrder_0__Mathlib_Tactic_initFn___closed__4_00___x40_Mathlib_Tactic_HigherOrder_2684385311____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_HigherOrder_0__Mathlib_Tactic_initFn___closed__5_00___x40_Mathlib_Tactic_HigherOrder_2684385311____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_HigherOrder_0__Mathlib_Tactic_initFn___closed__6_00___x40_Mathlib_Tactic_HigherOrder_2684385311____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_HigherOrder_0__Mathlib_Tactic_initFn___closed__7_00___x40_Mathlib_Tactic_HigherOrder_2684385311____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_HigherOrder_0__Mathlib_Tactic_initFn___closed__7_00___x40_Mathlib_Tactic_HigherOrder_2684385311____hygCtx___hyg_2__value;
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_HigherOrder_0__Mathlib_Tactic_initFn___closed__8_00___x40_Mathlib_Tactic_HigherOrder_2684385311____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Tactic_higherOrderGetParam___boxed, .m_arity = 5, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_HigherOrder_0__Mathlib_Tactic_initFn___closed__8_00___x40_Mathlib_Tactic_HigherOrder_2684385311____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_HigherOrder_0__Mathlib_Tactic_initFn___closed__8_00___x40_Mathlib_Tactic_HigherOrder_2684385311____hygCtx___hyg_2__value;
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_HigherOrder_0__Mathlib_Tactic_initFn___closed__9_00___x40_Mathlib_Tactic_HigherOrder_2684385311____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_HigherOrder_0__Mathlib_Tactic_initFn___lam__1_00___x40_Mathlib_Tactic_HigherOrder_2684385311____hygCtx___hyg_2____boxed, .m_arity = 4, .m_num_fixed = 1, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_HigherOrder_0__Mathlib_Tactic_initFn___closed__9_00___x40_Mathlib_Tactic_HigherOrder_2684385311____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_HigherOrder_0__Mathlib_Tactic_initFn___closed__9_00___x40_Mathlib_Tactic_HigherOrder_2684385311____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_HigherOrder_0__Mathlib_Tactic_initFn___closed__10_00___x40_Mathlib_Tactic_HigherOrder_2684385311____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 8, .m_other = 4, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_HigherOrder_0__Mathlib_Tactic_initFn___closed__7_00___x40_Mathlib_Tactic_HigherOrder_2684385311____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_HigherOrder_0__Mathlib_Tactic_initFn___closed__8_00___x40_Mathlib_Tactic_HigherOrder_2684385311____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_HigherOrder_0__Mathlib_Tactic_initFn___closed__0_00___x40_Mathlib_Tactic_HigherOrder_2684385311____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_HigherOrder_0__Mathlib_Tactic_initFn___closed__9_00___x40_Mathlib_Tactic_HigherOrder_2684385311____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_HigherOrder_0__Mathlib_Tactic_initFn___closed__10_00___x40_Mathlib_Tactic_HigherOrder_2684385311____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_HigherOrder_0__Mathlib_Tactic_initFn___closed__10_00___x40_Mathlib_Tactic_HigherOrder_2684385311____hygCtx___hyg_2__value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_HigherOrder_0__Mathlib_Tactic_initFn_00___x40_Mathlib_Tactic_HigherOrder_2684385311____hygCtx___hyg_2_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_HigherOrder_0__Mathlib_Tactic_initFn_00___x40_Mathlib_Tactic_HigherOrder_2684385311____hygCtx___hyg_2____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_higherOrderAttr;
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_mkComp_spec__0_spec__0(lean_object* v_msgData_46_, lean_object* v___y_47_, lean_object* v___y_48_, lean_object* v___y_49_, lean_object* v___y_50_){
_start:
{
lean_object* v___x_52_; lean_object* v_env_53_; lean_object* v___x_54_; lean_object* v_mctx_55_; lean_object* v_lctx_56_; lean_object* v_options_57_; lean_object* v___x_58_; lean_object* v___x_59_; lean_object* v___x_60_; 
v___x_52_ = lean_st_ref_get(v___y_50_);
v_env_53_ = lean_ctor_get(v___x_52_, 0);
lean_inc_ref(v_env_53_);
lean_dec(v___x_52_);
v___x_54_ = lean_st_ref_get(v___y_48_);
v_mctx_55_ = lean_ctor_get(v___x_54_, 0);
lean_inc_ref(v_mctx_55_);
lean_dec(v___x_54_);
v_lctx_56_ = lean_ctor_get(v___y_47_, 2);
v_options_57_ = lean_ctor_get(v___y_49_, 2);
lean_inc_ref(v_options_57_);
lean_inc_ref(v_lctx_56_);
v___x_58_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_58_, 0, v_env_53_);
lean_ctor_set(v___x_58_, 1, v_mctx_55_);
lean_ctor_set(v___x_58_, 2, v_lctx_56_);
lean_ctor_set(v___x_58_, 3, v_options_57_);
v___x_59_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_59_, 0, v___x_58_);
lean_ctor_set(v___x_59_, 1, v_msgData_46_);
v___x_60_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_60_, 0, v___x_59_);
return v___x_60_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_mkComp_spec__0_spec__0___boxed(lean_object* v_msgData_61_, lean_object* v___y_62_, lean_object* v___y_63_, lean_object* v___y_64_, lean_object* v___y_65_, lean_object* v___y_66_){
_start:
{
lean_object* v_res_67_; 
v_res_67_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_mkComp_spec__0_spec__0(v_msgData_61_, v___y_62_, v___y_63_, v___y_64_, v___y_65_);
lean_dec(v___y_65_);
lean_dec_ref(v___y_64_);
lean_dec(v___y_63_);
lean_dec_ref(v___y_62_);
return v_res_67_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_mkComp_spec__0___redArg(lean_object* v_msg_68_, lean_object* v___y_69_, lean_object* v___y_70_, lean_object* v___y_71_, lean_object* v___y_72_){
_start:
{
lean_object* v_ref_74_; lean_object* v___x_75_; lean_object* v_a_76_; lean_object* v___x_78_; uint8_t v_isShared_79_; uint8_t v_isSharedCheck_84_; 
v_ref_74_ = lean_ctor_get(v___y_71_, 5);
v___x_75_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_mkComp_spec__0_spec__0(v_msg_68_, v___y_69_, v___y_70_, v___y_71_, v___y_72_);
v_a_76_ = lean_ctor_get(v___x_75_, 0);
v_isSharedCheck_84_ = !lean_is_exclusive(v___x_75_);
if (v_isSharedCheck_84_ == 0)
{
v___x_78_ = v___x_75_;
v_isShared_79_ = v_isSharedCheck_84_;
goto v_resetjp_77_;
}
else
{
lean_inc(v_a_76_);
lean_dec(v___x_75_);
v___x_78_ = lean_box(0);
v_isShared_79_ = v_isSharedCheck_84_;
goto v_resetjp_77_;
}
v_resetjp_77_:
{
lean_object* v___x_80_; lean_object* v___x_82_; 
lean_inc(v_ref_74_);
v___x_80_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_80_, 0, v_ref_74_);
lean_ctor_set(v___x_80_, 1, v_a_76_);
if (v_isShared_79_ == 0)
{
lean_ctor_set_tag(v___x_78_, 1);
lean_ctor_set(v___x_78_, 0, v___x_80_);
v___x_82_ = v___x_78_;
goto v_reusejp_81_;
}
else
{
lean_object* v_reuseFailAlloc_83_; 
v_reuseFailAlloc_83_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_83_, 0, v___x_80_);
v___x_82_ = v_reuseFailAlloc_83_;
goto v_reusejp_81_;
}
v_reusejp_81_:
{
return v___x_82_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_mkComp_spec__0___redArg___boxed(lean_object* v_msg_85_, lean_object* v___y_86_, lean_object* v___y_87_, lean_object* v___y_88_, lean_object* v___y_89_, lean_object* v___y_90_){
_start:
{
lean_object* v_res_91_; 
v_res_91_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_mkComp_spec__0___redArg(v_msg_85_, v___y_86_, v___y_87_, v___y_88_, v___y_89_);
lean_dec(v___y_89_);
lean_dec_ref(v___y_88_);
lean_dec(v___y_87_);
lean_dec_ref(v___y_86_);
return v_res_91_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_mkComp___closed__6(void){
_start:
{
lean_object* v___x_101_; lean_object* v___x_102_; 
v___x_101_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_mkComp___closed__5));
v___x_102_ = l_Lean_stringToMessageData(v___x_101_);
return v___x_102_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_mkComp___closed__8(void){
_start:
{
lean_object* v___x_104_; lean_object* v___x_105_; 
v___x_104_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_mkComp___closed__7));
v___x_105_ = l_Lean_stringToMessageData(v___x_104_);
return v___x_105_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_mkComp(lean_object* v_v_106_, lean_object* v_x_107_, lean_object* v_a_108_, lean_object* v_a_109_, lean_object* v_a_110_, lean_object* v_a_111_){
_start:
{
if (lean_obj_tag(v_x_107_) == 5)
{
lean_object* v_fn_128_; lean_object* v_arg_129_; lean_object* v___y_131_; lean_object* v___y_132_; lean_object* v___y_133_; lean_object* v___y_134_; uint8_t v___x_143_; 
v_fn_128_ = lean_ctor_get(v_x_107_, 0);
lean_inc_ref(v_fn_128_);
v_arg_129_ = lean_ctor_get(v_x_107_, 1);
lean_inc_ref(v_arg_129_);
lean_dec_ref_known(v_x_107_, 2);
v___x_143_ = lean_expr_equal(v_arg_129_, v_v_106_);
if (v___x_143_ == 0)
{
uint8_t v___x_144_; 
lean_inc_ref(v_v_106_);
v___x_144_ = l_Lean_Expr_occurs(v_v_106_, v_fn_128_);
if (v___x_144_ == 0)
{
v___y_131_ = v_a_108_;
v___y_132_ = v_a_109_;
v___y_133_ = v_a_110_;
v___y_134_ = v_a_111_;
goto v___jp_130_;
}
else
{
lean_object* v___x_145_; lean_object* v___x_146_; lean_object* v_a_147_; lean_object* v___x_149_; uint8_t v_isShared_150_; uint8_t v_isSharedCheck_154_; 
lean_dec_ref(v_arg_129_);
lean_dec_ref(v_fn_128_);
lean_dec_ref(v_v_106_);
v___x_145_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_mkComp___closed__6, &lp_mathlib_Mathlib_Tactic_mkComp___closed__6_once, _init_lp_mathlib_Mathlib_Tactic_mkComp___closed__6);
v___x_146_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_mkComp_spec__0___redArg(v___x_145_, v_a_108_, v_a_109_, v_a_110_, v_a_111_);
v_a_147_ = lean_ctor_get(v___x_146_, 0);
v_isSharedCheck_154_ = !lean_is_exclusive(v___x_146_);
if (v_isSharedCheck_154_ == 0)
{
v___x_149_ = v___x_146_;
v_isShared_150_ = v_isSharedCheck_154_;
goto v_resetjp_148_;
}
else
{
lean_inc(v_a_147_);
lean_dec(v___x_146_);
v___x_149_ = lean_box(0);
v_isShared_150_ = v_isSharedCheck_154_;
goto v_resetjp_148_;
}
v_resetjp_148_:
{
lean_object* v___x_152_; 
if (v_isShared_150_ == 0)
{
v___x_152_ = v___x_149_;
goto v_reusejp_151_;
}
else
{
lean_object* v_reuseFailAlloc_153_; 
v_reuseFailAlloc_153_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_153_, 0, v_a_147_);
v___x_152_ = v_reuseFailAlloc_153_;
goto v_reusejp_151_;
}
v_reusejp_151_:
{
return v___x_152_;
}
}
}
}
else
{
lean_object* v___x_155_; 
lean_dec_ref(v_arg_129_);
lean_dec_ref(v_v_106_);
v___x_155_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_155_, 0, v_fn_128_);
return v___x_155_;
}
v___jp_130_:
{
lean_object* v___x_135_; 
v___x_135_ = lp_mathlib_Mathlib_Tactic_mkComp(v_v_106_, v_arg_129_, v___y_131_, v___y_132_, v___y_133_, v___y_134_);
if (lean_obj_tag(v___x_135_) == 0)
{
lean_object* v_a_136_; lean_object* v___x_137_; lean_object* v___x_138_; lean_object* v___x_139_; lean_object* v___x_140_; lean_object* v___x_141_; lean_object* v___x_142_; 
v_a_136_ = lean_ctor_get(v___x_135_, 0);
lean_inc(v_a_136_);
lean_dec_ref_known(v___x_135_, 1);
v___x_137_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_mkComp___closed__4));
v___x_138_ = lean_unsigned_to_nat(2u);
v___x_139_ = lean_mk_empty_array_with_capacity(v___x_138_);
v___x_140_ = lean_array_push(v___x_139_, v_fn_128_);
v___x_141_ = lean_array_push(v___x_140_, v_a_136_);
v___x_142_ = l_Lean_Meta_mkAppM(v___x_137_, v___x_141_, v___y_131_, v___y_132_, v___y_133_, v___y_134_);
return v___x_142_;
}
else
{
lean_dec_ref(v_fn_128_);
return v___x_135_;
}
}
}
else
{
uint8_t v___x_156_; 
v___x_156_ = lean_expr_equal(v_x_107_, v_v_106_);
lean_dec_ref(v_v_106_);
if (v___x_156_ == 0)
{
lean_object* v___x_157_; lean_object* v___x_158_; lean_object* v_a_159_; lean_object* v___x_161_; uint8_t v_isShared_162_; uint8_t v_isSharedCheck_166_; 
lean_dec_ref(v_x_107_);
v___x_157_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_mkComp___closed__8, &lp_mathlib_Mathlib_Tactic_mkComp___closed__8_once, _init_lp_mathlib_Mathlib_Tactic_mkComp___closed__8);
v___x_158_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_mkComp_spec__0___redArg(v___x_157_, v_a_108_, v_a_109_, v_a_110_, v_a_111_);
v_a_159_ = lean_ctor_get(v___x_158_, 0);
v_isSharedCheck_166_ = !lean_is_exclusive(v___x_158_);
if (v_isSharedCheck_166_ == 0)
{
v___x_161_ = v___x_158_;
v_isShared_162_ = v_isSharedCheck_166_;
goto v_resetjp_160_;
}
else
{
lean_inc(v_a_159_);
lean_dec(v___x_158_);
v___x_161_ = lean_box(0);
v_isShared_162_ = v_isSharedCheck_166_;
goto v_resetjp_160_;
}
v_resetjp_160_:
{
lean_object* v___x_164_; 
if (v_isShared_162_ == 0)
{
v___x_164_ = v___x_161_;
goto v_reusejp_163_;
}
else
{
lean_object* v_reuseFailAlloc_165_; 
v_reuseFailAlloc_165_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_165_, 0, v_a_159_);
v___x_164_ = v_reuseFailAlloc_165_;
goto v_reusejp_163_;
}
v_reusejp_163_:
{
return v___x_164_;
}
}
}
else
{
goto v___jp_113_;
}
}
v___jp_113_:
{
lean_object* v___x_114_; 
lean_inc(v_a_111_);
lean_inc_ref(v_a_110_);
lean_inc(v_a_109_);
lean_inc_ref(v_a_108_);
v___x_114_ = lean_infer_type(v_x_107_, v_a_108_, v_a_109_, v_a_110_, v_a_111_);
if (lean_obj_tag(v___x_114_) == 0)
{
lean_object* v_a_115_; lean_object* v___x_117_; uint8_t v_isShared_118_; uint8_t v_isSharedCheck_127_; 
v_a_115_ = lean_ctor_get(v___x_114_, 0);
v_isSharedCheck_127_ = !lean_is_exclusive(v___x_114_);
if (v_isSharedCheck_127_ == 0)
{
v___x_117_ = v___x_114_;
v_isShared_118_ = v_isSharedCheck_127_;
goto v_resetjp_116_;
}
else
{
lean_inc(v_a_115_);
lean_dec(v___x_114_);
v___x_117_ = lean_box(0);
v_isShared_118_ = v_isSharedCheck_127_;
goto v_resetjp_116_;
}
v_resetjp_116_:
{
lean_object* v___x_119_; lean_object* v___x_121_; 
v___x_119_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_mkComp___closed__1));
if (v_isShared_118_ == 0)
{
lean_ctor_set_tag(v___x_117_, 1);
v___x_121_ = v___x_117_;
goto v_reusejp_120_;
}
else
{
lean_object* v_reuseFailAlloc_126_; 
v_reuseFailAlloc_126_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_126_, 0, v_a_115_);
v___x_121_ = v_reuseFailAlloc_126_;
goto v_reusejp_120_;
}
v_reusejp_120_:
{
lean_object* v___x_122_; lean_object* v___x_123_; lean_object* v___x_124_; lean_object* v___x_125_; 
v___x_122_ = lean_unsigned_to_nat(1u);
v___x_123_ = lean_mk_empty_array_with_capacity(v___x_122_);
v___x_124_ = lean_array_push(v___x_123_, v___x_121_);
v___x_125_ = l_Lean_Meta_mkAppOptM(v___x_119_, v___x_124_, v_a_108_, v_a_109_, v_a_110_, v_a_111_);
return v___x_125_;
}
}
}
else
{
return v___x_114_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_mkComp___boxed(lean_object* v_v_167_, lean_object* v_x_168_, lean_object* v_a_169_, lean_object* v_a_170_, lean_object* v_a_171_, lean_object* v_a_172_, lean_object* v_a_173_){
_start:
{
lean_object* v_res_174_; 
v_res_174_ = lp_mathlib_Mathlib_Tactic_mkComp(v_v_167_, v_x_168_, v_a_169_, v_a_170_, v_a_171_, v_a_172_);
lean_dec(v_a_172_);
lean_dec_ref(v_a_171_);
lean_dec(v_a_170_);
lean_dec_ref(v_a_169_);
return v_res_174_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_mkComp_spec__0(lean_object* v_00_u03b1_175_, lean_object* v_msg_176_, lean_object* v___y_177_, lean_object* v___y_178_, lean_object* v___y_179_, lean_object* v___y_180_){
_start:
{
lean_object* v___x_182_; 
v___x_182_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_mkComp_spec__0___redArg(v_msg_176_, v___y_177_, v___y_178_, v___y_179_, v___y_180_);
return v___x_182_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_mkComp_spec__0___boxed(lean_object* v_00_u03b1_183_, lean_object* v_msg_184_, lean_object* v___y_185_, lean_object* v___y_186_, lean_object* v___y_187_, lean_object* v___y_188_, lean_object* v___y_189_){
_start:
{
lean_object* v_res_190_; 
v_res_190_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_mkComp_spec__0(v_00_u03b1_183_, v_msg_184_, v___y_185_, v___y_186_, v___y_187_, v___y_188_);
lean_dec(v___y_188_);
lean_dec_ref(v___y_187_);
lean_dec(v___y_186_);
lean_dec_ref(v___y_185_);
return v_res_190_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Mathlib_Tactic_mkHigherOrderType_spec__0___redArg___lam__0(lean_object* v_k_191_, lean_object* v_b_192_, lean_object* v___y_193_, lean_object* v___y_194_, lean_object* v___y_195_, lean_object* v___y_196_){
_start:
{
lean_object* v___x_198_; 
lean_inc(v___y_196_);
lean_inc_ref(v___y_195_);
lean_inc(v___y_194_);
lean_inc_ref(v___y_193_);
v___x_198_ = lean_apply_6(v_k_191_, v_b_192_, v___y_193_, v___y_194_, v___y_195_, v___y_196_, lean_box(0));
return v___x_198_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Mathlib_Tactic_mkHigherOrderType_spec__0___redArg___lam__0___boxed(lean_object* v_k_199_, lean_object* v_b_200_, lean_object* v___y_201_, lean_object* v___y_202_, lean_object* v___y_203_, lean_object* v___y_204_, lean_object* v___y_205_){
_start:
{
lean_object* v_res_206_; 
v_res_206_ = lp_mathlib_Lean_Meta_withLocalDecl___at___00Mathlib_Tactic_mkHigherOrderType_spec__0___redArg___lam__0(v_k_199_, v_b_200_, v___y_201_, v___y_202_, v___y_203_, v___y_204_);
lean_dec(v___y_204_);
lean_dec_ref(v___y_203_);
lean_dec(v___y_202_);
lean_dec_ref(v___y_201_);
return v_res_206_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Mathlib_Tactic_mkHigherOrderType_spec__0___redArg(lean_object* v_name_207_, uint8_t v_bi_208_, lean_object* v_type_209_, lean_object* v_k_210_, uint8_t v_kind_211_, lean_object* v___y_212_, lean_object* v___y_213_, lean_object* v___y_214_, lean_object* v___y_215_){
_start:
{
lean_object* v___f_217_; lean_object* v___x_218_; 
v___f_217_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Meta_withLocalDecl___at___00Mathlib_Tactic_mkHigherOrderType_spec__0___redArg___lam__0___boxed), 7, 1);
lean_closure_set(v___f_217_, 0, v_k_210_);
v___x_218_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withLocalDeclImp(lean_box(0), v_name_207_, v_bi_208_, v_type_209_, v___f_217_, v_kind_211_, v___y_212_, v___y_213_, v___y_214_, v___y_215_);
if (lean_obj_tag(v___x_218_) == 0)
{
lean_object* v_a_219_; lean_object* v___x_221_; uint8_t v_isShared_222_; uint8_t v_isSharedCheck_226_; 
v_a_219_ = lean_ctor_get(v___x_218_, 0);
v_isSharedCheck_226_ = !lean_is_exclusive(v___x_218_);
if (v_isSharedCheck_226_ == 0)
{
v___x_221_ = v___x_218_;
v_isShared_222_ = v_isSharedCheck_226_;
goto v_resetjp_220_;
}
else
{
lean_inc(v_a_219_);
lean_dec(v___x_218_);
v___x_221_ = lean_box(0);
v_isShared_222_ = v_isSharedCheck_226_;
goto v_resetjp_220_;
}
v_resetjp_220_:
{
lean_object* v___x_224_; 
if (v_isShared_222_ == 0)
{
v___x_224_ = v___x_221_;
goto v_reusejp_223_;
}
else
{
lean_object* v_reuseFailAlloc_225_; 
v_reuseFailAlloc_225_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_225_, 0, v_a_219_);
v___x_224_ = v_reuseFailAlloc_225_;
goto v_reusejp_223_;
}
v_reusejp_223_:
{
return v___x_224_;
}
}
}
else
{
lean_object* v_a_227_; lean_object* v___x_229_; uint8_t v_isShared_230_; uint8_t v_isSharedCheck_234_; 
v_a_227_ = lean_ctor_get(v___x_218_, 0);
v_isSharedCheck_234_ = !lean_is_exclusive(v___x_218_);
if (v_isSharedCheck_234_ == 0)
{
v___x_229_ = v___x_218_;
v_isShared_230_ = v_isSharedCheck_234_;
goto v_resetjp_228_;
}
else
{
lean_inc(v_a_227_);
lean_dec(v___x_218_);
v___x_229_ = lean_box(0);
v_isShared_230_ = v_isSharedCheck_234_;
goto v_resetjp_228_;
}
v_resetjp_228_:
{
lean_object* v___x_232_; 
if (v_isShared_230_ == 0)
{
v___x_232_ = v___x_229_;
goto v_reusejp_231_;
}
else
{
lean_object* v_reuseFailAlloc_233_; 
v_reuseFailAlloc_233_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_233_, 0, v_a_227_);
v___x_232_ = v_reuseFailAlloc_233_;
goto v_reusejp_231_;
}
v_reusejp_231_:
{
return v___x_232_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Mathlib_Tactic_mkHigherOrderType_spec__0___redArg___boxed(lean_object* v_name_235_, lean_object* v_bi_236_, lean_object* v_type_237_, lean_object* v_k_238_, lean_object* v_kind_239_, lean_object* v___y_240_, lean_object* v___y_241_, lean_object* v___y_242_, lean_object* v___y_243_, lean_object* v___y_244_){
_start:
{
uint8_t v_bi_boxed_245_; uint8_t v_kind_boxed_246_; lean_object* v_res_247_; 
v_bi_boxed_245_ = lean_unbox(v_bi_236_);
v_kind_boxed_246_ = lean_unbox(v_kind_239_);
v_res_247_ = lp_mathlib_Lean_Meta_withLocalDecl___at___00Mathlib_Tactic_mkHigherOrderType_spec__0___redArg(v_name_235_, v_bi_boxed_245_, v_type_237_, v_k_238_, v_kind_boxed_246_, v___y_240_, v___y_241_, v___y_242_, v___y_243_);
lean_dec(v___y_243_);
lean_dec_ref(v___y_242_);
lean_dec(v___y_241_);
lean_dec_ref(v___y_240_);
return v_res_247_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Mathlib_Tactic_mkHigherOrderType_spec__0(lean_object* v_00_u03b1_248_, lean_object* v_name_249_, uint8_t v_bi_250_, lean_object* v_type_251_, lean_object* v_k_252_, uint8_t v_kind_253_, lean_object* v___y_254_, lean_object* v___y_255_, lean_object* v___y_256_, lean_object* v___y_257_){
_start:
{
lean_object* v___x_259_; 
v___x_259_ = lp_mathlib_Lean_Meta_withLocalDecl___at___00Mathlib_Tactic_mkHigherOrderType_spec__0___redArg(v_name_249_, v_bi_250_, v_type_251_, v_k_252_, v_kind_253_, v___y_254_, v___y_255_, v___y_256_, v___y_257_);
return v___x_259_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Mathlib_Tactic_mkHigherOrderType_spec__0___boxed(lean_object* v_00_u03b1_260_, lean_object* v_name_261_, lean_object* v_bi_262_, lean_object* v_type_263_, lean_object* v_k_264_, lean_object* v_kind_265_, lean_object* v___y_266_, lean_object* v___y_267_, lean_object* v___y_268_, lean_object* v___y_269_, lean_object* v___y_270_){
_start:
{
uint8_t v_bi_boxed_271_; uint8_t v_kind_boxed_272_; lean_object* v_res_273_; 
v_bi_boxed_271_ = lean_unbox(v_bi_262_);
v_kind_boxed_272_ = lean_unbox(v_kind_265_);
v_res_273_ = lp_mathlib_Lean_Meta_withLocalDecl___at___00Mathlib_Tactic_mkHigherOrderType_spec__0(v_00_u03b1_260_, v_name_261_, v_bi_boxed_271_, v_type_263_, v_k_264_, v_kind_boxed_272_, v___y_266_, v___y_267_, v___y_268_, v___y_269_);
lean_dec(v___y_269_);
lean_dec_ref(v___y_268_);
lean_dec(v___y_267_);
lean_dec_ref(v___y_266_);
return v_res_273_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_mkHigherOrderType___closed__1(void){
_start:
{
lean_object* v___x_275_; lean_object* v___x_276_; 
v___x_275_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_mkHigherOrderType___closed__0));
v___x_276_ = l_Lean_stringToMessageData(v___x_275_);
return v___x_276_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_mkHigherOrderType___lam__0___closed__1(void){
_start:
{
lean_object* v___x_278_; lean_object* v___x_279_; 
v___x_278_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_mkHigherOrderType___lam__0___closed__0));
v___x_279_ = l_Lean_stringToMessageData(v___x_278_);
return v___x_279_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_mkHigherOrderType___lam__0(lean_object* v_e_280_, uint8_t v___x_281_, lean_object* v_fvar_282_, lean_object* v___y_283_, lean_object* v___y_284_, lean_object* v___y_285_, lean_object* v___y_286_){
_start:
{
lean_object* v___x_288_; lean_object* v_body_289_; uint8_t v___x_290_; 
v___x_288_ = l_Lean_Expr_bindingBody_x21(v_e_280_);
v_body_289_ = lean_expr_instantiate1(v___x_288_, v_fvar_282_);
lean_dec_ref(v___x_288_);
v___x_290_ = l_Lean_Expr_isForall(v_body_289_);
if (v___x_290_ == 0)
{
lean_object* v___x_291_; 
lean_inc_ref(v_body_289_);
v___x_291_ = l_Lean_Meta_matchEq_x3f(v_body_289_, v___y_283_, v___y_284_, v___y_285_, v___y_286_);
if (lean_obj_tag(v___x_291_) == 0)
{
lean_object* v_a_292_; 
v_a_292_ = lean_ctor_get(v___x_291_, 0);
lean_inc(v_a_292_);
lean_dec_ref_known(v___x_291_, 1);
if (lean_obj_tag(v_a_292_) == 1)
{
lean_object* v_val_293_; lean_object* v_snd_294_; lean_object* v_fst_295_; lean_object* v_snd_296_; lean_object* v___x_297_; 
lean_dec_ref(v_body_289_);
v_val_293_ = lean_ctor_get(v_a_292_, 0);
lean_inc(v_val_293_);
lean_dec_ref_known(v_a_292_, 1);
v_snd_294_ = lean_ctor_get(v_val_293_, 1);
lean_inc(v_snd_294_);
lean_dec(v_val_293_);
v_fst_295_ = lean_ctor_get(v_snd_294_, 0);
lean_inc(v_fst_295_);
v_snd_296_ = lean_ctor_get(v_snd_294_, 1);
lean_inc(v_snd_296_);
lean_dec(v_snd_294_);
lean_inc_ref(v_fvar_282_);
v___x_297_ = lp_mathlib_Mathlib_Tactic_mkComp(v_fvar_282_, v_fst_295_, v___y_283_, v___y_284_, v___y_285_, v___y_286_);
if (lean_obj_tag(v___x_297_) == 0)
{
lean_object* v_a_298_; lean_object* v___x_299_; 
v_a_298_ = lean_ctor_get(v___x_297_, 0);
lean_inc(v_a_298_);
lean_dec_ref_known(v___x_297_, 1);
v___x_299_ = lp_mathlib_Mathlib_Tactic_mkComp(v_fvar_282_, v_snd_296_, v___y_283_, v___y_284_, v___y_285_, v___y_286_);
if (lean_obj_tag(v___x_299_) == 0)
{
lean_object* v_a_300_; lean_object* v___x_301_; 
v_a_300_ = lean_ctor_get(v___x_299_, 0);
lean_inc(v_a_300_);
lean_dec_ref_known(v___x_299_, 1);
v___x_301_ = l_Lean_Meta_mkEq(v_a_298_, v_a_300_, v___y_283_, v___y_284_, v___y_285_, v___y_286_);
return v___x_301_;
}
else
{
lean_dec(v_a_298_);
return v___x_299_;
}
}
else
{
lean_dec(v_snd_296_);
lean_dec_ref(v_fvar_282_);
return v___x_297_;
}
}
else
{
lean_object* v___x_302_; 
lean_dec(v_a_292_);
lean_dec_ref(v_fvar_282_);
v___x_302_ = l_Lean_Meta_ppExpr(v_body_289_, v___y_283_, v___y_284_, v___y_285_, v___y_286_);
if (lean_obj_tag(v___x_302_) == 0)
{
lean_object* v_a_303_; lean_object* v___x_304_; lean_object* v___x_305_; lean_object* v___x_306_; lean_object* v___x_307_; 
v_a_303_ = lean_ctor_get(v___x_302_, 0);
lean_inc(v_a_303_);
lean_dec_ref_known(v___x_302_, 1);
v___x_304_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_mkHigherOrderType___lam__0___closed__1, &lp_mathlib_Mathlib_Tactic_mkHigherOrderType___lam__0___closed__1_once, _init_lp_mathlib_Mathlib_Tactic_mkHigherOrderType___lam__0___closed__1);
v___x_305_ = l_Lean_MessageData_ofFormat(v_a_303_);
v___x_306_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_306_, 0, v___x_304_);
lean_ctor_set(v___x_306_, 1, v___x_305_);
v___x_307_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_mkComp_spec__0___redArg(v___x_306_, v___y_283_, v___y_284_, v___y_285_, v___y_286_);
return v___x_307_;
}
else
{
lean_object* v_a_308_; lean_object* v___x_310_; uint8_t v_isShared_311_; uint8_t v_isSharedCheck_315_; 
v_a_308_ = lean_ctor_get(v___x_302_, 0);
v_isSharedCheck_315_ = !lean_is_exclusive(v___x_302_);
if (v_isSharedCheck_315_ == 0)
{
v___x_310_ = v___x_302_;
v_isShared_311_ = v_isSharedCheck_315_;
goto v_resetjp_309_;
}
else
{
lean_inc(v_a_308_);
lean_dec(v___x_302_);
v___x_310_ = lean_box(0);
v_isShared_311_ = v_isSharedCheck_315_;
goto v_resetjp_309_;
}
v_resetjp_309_:
{
lean_object* v___x_313_; 
if (v_isShared_311_ == 0)
{
v___x_313_ = v___x_310_;
goto v_reusejp_312_;
}
else
{
lean_object* v_reuseFailAlloc_314_; 
v_reuseFailAlloc_314_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_314_, 0, v_a_308_);
v___x_313_ = v_reuseFailAlloc_314_;
goto v_reusejp_312_;
}
v_reusejp_312_:
{
return v___x_313_;
}
}
}
}
}
else
{
lean_object* v_a_316_; lean_object* v___x_318_; uint8_t v_isShared_319_; uint8_t v_isSharedCheck_323_; 
lean_dec_ref(v_body_289_);
lean_dec_ref(v_fvar_282_);
v_a_316_ = lean_ctor_get(v___x_291_, 0);
v_isSharedCheck_323_ = !lean_is_exclusive(v___x_291_);
if (v_isSharedCheck_323_ == 0)
{
v___x_318_ = v___x_291_;
v_isShared_319_ = v_isSharedCheck_323_;
goto v_resetjp_317_;
}
else
{
lean_inc(v_a_316_);
lean_dec(v___x_291_);
v___x_318_ = lean_box(0);
v_isShared_319_ = v_isSharedCheck_323_;
goto v_resetjp_317_;
}
v_resetjp_317_:
{
lean_object* v___x_321_; 
if (v_isShared_319_ == 0)
{
v___x_321_ = v___x_318_;
goto v_reusejp_320_;
}
else
{
lean_object* v_reuseFailAlloc_322_; 
v_reuseFailAlloc_322_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_322_, 0, v_a_316_);
v___x_321_ = v_reuseFailAlloc_322_;
goto v_reusejp_320_;
}
v_reusejp_320_:
{
return v___x_321_;
}
}
}
}
else
{
lean_object* v___x_324_; 
v___x_324_ = lp_mathlib_Mathlib_Tactic_mkHigherOrderType(v_body_289_, v___y_283_, v___y_284_, v___y_285_, v___y_286_);
if (lean_obj_tag(v___x_324_) == 0)
{
lean_object* v_a_325_; lean_object* v___x_326_; lean_object* v___x_327_; lean_object* v___x_328_; uint8_t v___x_329_; lean_object* v___x_330_; 
v_a_325_ = lean_ctor_get(v___x_324_, 0);
lean_inc(v_a_325_);
lean_dec_ref_known(v___x_324_, 1);
v___x_326_ = lean_unsigned_to_nat(1u);
v___x_327_ = lean_mk_empty_array_with_capacity(v___x_326_);
v___x_328_ = lean_array_push(v___x_327_, v_fvar_282_);
v___x_329_ = 0;
v___x_330_ = l_Lean_Meta_mkForallFVars(v___x_328_, v_a_325_, v___x_329_, v___x_290_, v___x_290_, v___x_281_, v___y_283_, v___y_284_, v___y_285_, v___y_286_);
lean_dec_ref(v___x_328_);
return v___x_330_;
}
else
{
lean_dec_ref(v_fvar_282_);
return v___x_324_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_mkHigherOrderType___lam__0___boxed(lean_object* v_e_331_, lean_object* v___x_332_, lean_object* v_fvar_333_, lean_object* v___y_334_, lean_object* v___y_335_, lean_object* v___y_336_, lean_object* v___y_337_, lean_object* v___y_338_){
_start:
{
uint8_t v___x_1456__boxed_339_; lean_object* v_res_340_; 
v___x_1456__boxed_339_ = lean_unbox(v___x_332_);
v_res_340_ = lp_mathlib_Mathlib_Tactic_mkHigherOrderType___lam__0(v_e_331_, v___x_1456__boxed_339_, v_fvar_333_, v___y_334_, v___y_335_, v___y_336_, v___y_337_);
lean_dec(v___y_337_);
lean_dec_ref(v___y_336_);
lean_dec(v___y_335_);
lean_dec_ref(v___y_334_);
lean_dec_ref(v_e_331_);
return v_res_340_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_mkHigherOrderType(lean_object* v_e_341_, lean_object* v_a_342_, lean_object* v_a_343_, lean_object* v_a_344_, lean_object* v_a_345_){
_start:
{
lean_object* v___y_348_; lean_object* v___y_349_; lean_object* v___y_350_; lean_object* v___y_351_; uint8_t v___x_359_; 
v___x_359_ = l_Lean_Expr_isForall(v_e_341_);
if (v___x_359_ == 0)
{
lean_object* v___x_360_; lean_object* v___x_361_; 
v___x_360_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_mkHigherOrderType___closed__1, &lp_mathlib_Mathlib_Tactic_mkHigherOrderType___closed__1_once, _init_lp_mathlib_Mathlib_Tactic_mkHigherOrderType___closed__1);
v___x_361_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_mkComp_spec__0___redArg(v___x_360_, v_a_342_, v_a_343_, v_a_344_, v_a_345_);
if (lean_obj_tag(v___x_361_) == 0)
{
lean_dec_ref_known(v___x_361_, 1);
v___y_348_ = v_a_342_;
v___y_349_ = v_a_343_;
v___y_350_ = v_a_344_;
v___y_351_ = v_a_345_;
goto v___jp_347_;
}
else
{
lean_object* v_a_362_; lean_object* v___x_364_; uint8_t v_isShared_365_; uint8_t v_isSharedCheck_369_; 
lean_dec_ref(v_e_341_);
v_a_362_ = lean_ctor_get(v___x_361_, 0);
v_isSharedCheck_369_ = !lean_is_exclusive(v___x_361_);
if (v_isSharedCheck_369_ == 0)
{
v___x_364_ = v___x_361_;
v_isShared_365_ = v_isSharedCheck_369_;
goto v_resetjp_363_;
}
else
{
lean_inc(v_a_362_);
lean_dec(v___x_361_);
v___x_364_ = lean_box(0);
v_isShared_365_ = v_isSharedCheck_369_;
goto v_resetjp_363_;
}
v_resetjp_363_:
{
lean_object* v___x_367_; 
if (v_isShared_365_ == 0)
{
v___x_367_ = v___x_364_;
goto v_reusejp_366_;
}
else
{
lean_object* v_reuseFailAlloc_368_; 
v_reuseFailAlloc_368_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_368_, 0, v_a_362_);
v___x_367_ = v_reuseFailAlloc_368_;
goto v_reusejp_366_;
}
v_reusejp_366_:
{
return v___x_367_;
}
}
}
}
else
{
v___y_348_ = v_a_342_;
v___y_349_ = v_a_343_;
v___y_350_ = v_a_344_;
v___y_351_ = v_a_345_;
goto v___jp_347_;
}
v___jp_347_:
{
lean_object* v___x_352_; uint8_t v___x_353_; lean_object* v___x_354_; lean_object* v___f_355_; lean_object* v___x_356_; uint8_t v___x_357_; lean_object* v___x_358_; 
v___x_352_ = l_Lean_Expr_bindingName_x21(v_e_341_);
v___x_353_ = l_Lean_Expr_binderInfo(v_e_341_);
v___x_354_ = lean_box(v___x_353_);
lean_inc_ref(v_e_341_);
v___f_355_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_mkHigherOrderType___lam__0___boxed), 8, 2);
lean_closure_set(v___f_355_, 0, v_e_341_);
lean_closure_set(v___f_355_, 1, v___x_354_);
v___x_356_ = l_Lean_Expr_bindingDomain_x21(v_e_341_);
lean_dec_ref(v_e_341_);
v___x_357_ = 0;
v___x_358_ = lp_mathlib_Lean_Meta_withLocalDecl___at___00Mathlib_Tactic_mkHigherOrderType_spec__0___redArg(v___x_352_, v___x_353_, v___x_356_, v___f_355_, v___x_357_, v___y_348_, v___y_349_, v___y_350_, v___y_351_);
return v___x_358_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_mkHigherOrderType___boxed(lean_object* v_e_370_, lean_object* v_a_371_, lean_object* v_a_372_, lean_object* v_a_373_, lean_object* v_a_374_, lean_object* v_a_375_){
_start:
{
lean_object* v_res_376_; 
v_res_376_ = lp_mathlib_Mathlib_Tactic_mkHigherOrderType(v_e_370_, v_a_371_, v_a_372_, v_a_373_, v_a_374_);
lean_dec(v_a_374_);
lean_dec_ref(v_a_373_);
lean_dec(v_a_372_);
lean_dec_ref(v_a_371_);
return v_res_376_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_higherOrderGetParam_spec__0___redArg___closed__0(void){
_start:
{
lean_object* v___x_377_; lean_object* v___x_378_; lean_object* v___x_379_; 
v___x_377_ = lean_box(0);
v___x_378_ = l_Lean_Elab_unsupportedSyntaxExceptionId;
v___x_379_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_379_, 0, v___x_378_);
lean_ctor_set(v___x_379_, 1, v___x_377_);
return v___x_379_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_higherOrderGetParam_spec__0___redArg(){
_start:
{
lean_object* v___x_381_; lean_object* v___x_382_; 
v___x_381_ = lean_obj_once(&lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_higherOrderGetParam_spec__0___redArg___closed__0, &lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_higherOrderGetParam_spec__0___redArg___closed__0_once, _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_higherOrderGetParam_spec__0___redArg___closed__0);
v___x_382_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_382_, 0, v___x_381_);
return v___x_382_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_higherOrderGetParam_spec__0___redArg___boxed(lean_object* v___y_383_){
_start:
{
lean_object* v_res_384_; 
v_res_384_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_higherOrderGetParam_spec__0___redArg();
return v_res_384_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_higherOrderGetParam_spec__0(lean_object* v_00_u03b1_385_, lean_object* v___y_386_, lean_object* v___y_387_){
_start:
{
lean_object* v___x_389_; 
v___x_389_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_higherOrderGetParam_spec__0___redArg();
return v___x_389_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_higherOrderGetParam_spec__0___boxed(lean_object* v_00_u03b1_390_, lean_object* v___y_391_, lean_object* v___y_392_, lean_object* v___y_393_){
_start:
{
lean_object* v_res_394_; 
v_res_394_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_higherOrderGetParam_spec__0(v_00_u03b1_390_, v___y_391_, v___y_392_);
lean_dec(v___y_392_);
lean_dec_ref(v___y_391_);
return v_res_394_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_higherOrderGetParam_spec__3___redArg(lean_object* v_e_395_, lean_object* v___y_396_){
_start:
{
uint8_t v___x_398_; 
v___x_398_ = l_Lean_Expr_hasMVar(v_e_395_);
if (v___x_398_ == 0)
{
lean_object* v___x_399_; 
v___x_399_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_399_, 0, v_e_395_);
return v___x_399_;
}
else
{
lean_object* v___x_400_; lean_object* v_mctx_401_; lean_object* v___x_402_; lean_object* v_fst_403_; lean_object* v_snd_404_; lean_object* v___x_405_; lean_object* v_cache_406_; lean_object* v_zetaDeltaFVarIds_407_; lean_object* v_postponed_408_; lean_object* v_diag_409_; lean_object* v___x_411_; uint8_t v_isShared_412_; uint8_t v_isSharedCheck_418_; 
v___x_400_ = lean_st_ref_get(v___y_396_);
v_mctx_401_ = lean_ctor_get(v___x_400_, 0);
lean_inc_ref(v_mctx_401_);
lean_dec(v___x_400_);
v___x_402_ = l_Lean_instantiateMVarsCore(v_mctx_401_, v_e_395_);
v_fst_403_ = lean_ctor_get(v___x_402_, 0);
lean_inc(v_fst_403_);
v_snd_404_ = lean_ctor_get(v___x_402_, 1);
lean_inc(v_snd_404_);
lean_dec_ref(v___x_402_);
v___x_405_ = lean_st_ref_take(v___y_396_);
v_cache_406_ = lean_ctor_get(v___x_405_, 1);
v_zetaDeltaFVarIds_407_ = lean_ctor_get(v___x_405_, 2);
v_postponed_408_ = lean_ctor_get(v___x_405_, 3);
v_diag_409_ = lean_ctor_get(v___x_405_, 4);
v_isSharedCheck_418_ = !lean_is_exclusive(v___x_405_);
if (v_isSharedCheck_418_ == 0)
{
lean_object* v_unused_419_; 
v_unused_419_ = lean_ctor_get(v___x_405_, 0);
lean_dec(v_unused_419_);
v___x_411_ = v___x_405_;
v_isShared_412_ = v_isSharedCheck_418_;
goto v_resetjp_410_;
}
else
{
lean_inc(v_diag_409_);
lean_inc(v_postponed_408_);
lean_inc(v_zetaDeltaFVarIds_407_);
lean_inc(v_cache_406_);
lean_dec(v___x_405_);
v___x_411_ = lean_box(0);
v_isShared_412_ = v_isSharedCheck_418_;
goto v_resetjp_410_;
}
v_resetjp_410_:
{
lean_object* v___x_414_; 
if (v_isShared_412_ == 0)
{
lean_ctor_set(v___x_411_, 0, v_snd_404_);
v___x_414_ = v___x_411_;
goto v_reusejp_413_;
}
else
{
lean_object* v_reuseFailAlloc_417_; 
v_reuseFailAlloc_417_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_417_, 0, v_snd_404_);
lean_ctor_set(v_reuseFailAlloc_417_, 1, v_cache_406_);
lean_ctor_set(v_reuseFailAlloc_417_, 2, v_zetaDeltaFVarIds_407_);
lean_ctor_set(v_reuseFailAlloc_417_, 3, v_postponed_408_);
lean_ctor_set(v_reuseFailAlloc_417_, 4, v_diag_409_);
v___x_414_ = v_reuseFailAlloc_417_;
goto v_reusejp_413_;
}
v_reusejp_413_:
{
lean_object* v___x_415_; lean_object* v___x_416_; 
v___x_415_ = lean_st_ref_set(v___y_396_, v___x_414_);
v___x_416_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_416_, 0, v_fst_403_);
return v___x_416_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_higherOrderGetParam_spec__3___redArg___boxed(lean_object* v_e_420_, lean_object* v___y_421_, lean_object* v___y_422_){
_start:
{
lean_object* v_res_423_; 
v_res_423_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_higherOrderGetParam_spec__3___redArg(v_e_420_, v___y_421_);
lean_dec(v___y_421_);
return v_res_423_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_higherOrderGetParam_spec__3(lean_object* v_e_424_, lean_object* v___y_425_, lean_object* v___y_426_, lean_object* v___y_427_, lean_object* v___y_428_, lean_object* v___y_429_, lean_object* v___y_430_){
_start:
{
lean_object* v___x_432_; 
v___x_432_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_higherOrderGetParam_spec__3___redArg(v_e_424_, v___y_428_);
return v___x_432_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_higherOrderGetParam_spec__3___boxed(lean_object* v_e_433_, lean_object* v___y_434_, lean_object* v___y_435_, lean_object* v___y_436_, lean_object* v___y_437_, lean_object* v___y_438_, lean_object* v___y_439_, lean_object* v___y_440_){
_start:
{
lean_object* v_res_441_; 
v_res_441_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_higherOrderGetParam_spec__3(v_e_433_, v___y_434_, v___y_435_, v___y_436_, v___y_437_, v___y_438_, v___y_439_);
lean_dec(v___y_439_);
lean_dec_ref(v___y_438_);
lean_dec(v___y_437_);
lean_dec_ref(v___y_436_);
lean_dec(v___y_435_);
lean_dec_ref(v___y_434_);
return v_res_441_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Tactic_higherOrderGetParam___lam__0(lean_object* v_x_442_){
_start:
{
uint8_t v___x_443_; 
v___x_443_ = 0;
return v___x_443_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_higherOrderGetParam___lam__0___boxed(lean_object* v_x_444_){
_start:
{
uint8_t v_res_445_; lean_object* v_r_446_; 
v_res_445_ = lp_mathlib_Mathlib_Tactic_higherOrderGetParam___lam__0(v_x_444_);
lean_dec(v_x_444_);
v_r_446_ = lean_box(v_res_445_);
return v_r_446_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Mathlib_Tactic_higherOrderGetParam_spec__6_spec__10_spec__12___redArg(lean_object* v_keys_447_, lean_object* v_i_448_, lean_object* v_k_449_){
_start:
{
uint8_t v___y_455_; lean_object* v___x_456_; uint8_t v___x_457_; 
v___x_456_ = lean_array_get_size(v_keys_447_);
v___x_457_ = lean_nat_dec_lt(v_i_448_, v___x_456_);
if (v___x_457_ == 0)
{
lean_dec(v_i_448_);
return v___x_457_;
}
else
{
lean_object* v_k_x27_458_; 
v_k_x27_458_ = lean_array_fget_borrowed(v_keys_447_, v_i_448_);
if (lean_obj_tag(v_k_449_) == 0)
{
if (lean_obj_tag(v_k_x27_458_) == 0)
{
lean_object* v_declName_459_; uint8_t v_inv_460_; lean_object* v_declName_461_; uint8_t v_inv_462_; uint8_t v___x_463_; 
v_declName_459_ = lean_ctor_get(v_k_449_, 0);
v_inv_460_ = lean_ctor_get_uint8(v_k_449_, sizeof(void*)*1 + 1);
v_declName_461_ = lean_ctor_get(v_k_x27_458_, 0);
v_inv_462_ = lean_ctor_get_uint8(v_k_x27_458_, sizeof(void*)*1 + 1);
v___x_463_ = lean_name_eq(v_declName_459_, v_declName_461_);
if (v___x_463_ == 0)
{
v___y_455_ = v___x_463_;
goto v___jp_454_;
}
else
{
if (v_inv_460_ == 0)
{
if (v_inv_462_ == 0)
{
v___y_455_ = v___x_463_;
goto v___jp_454_;
}
else
{
goto v___jp_450_;
}
}
else
{
v___y_455_ = v_inv_462_;
goto v___jp_454_;
}
}
}
else
{
goto v___jp_450_;
}
}
else
{
if (lean_obj_tag(v_k_x27_458_) == 0)
{
goto v___jp_450_;
}
else
{
lean_object* v___x_464_; lean_object* v___x_465_; uint8_t v___x_466_; 
v___x_464_ = l_Lean_Meta_Origin_key(v_k_449_);
v___x_465_ = l_Lean_Meta_Origin_key(v_k_x27_458_);
v___x_466_ = lean_name_eq(v___x_464_, v___x_465_);
lean_dec(v___x_465_);
lean_dec(v___x_464_);
v___y_455_ = v___x_466_;
goto v___jp_454_;
}
}
}
v___jp_450_:
{
lean_object* v___x_451_; lean_object* v___x_452_; 
v___x_451_ = lean_unsigned_to_nat(1u);
v___x_452_ = lean_nat_add(v_i_448_, v___x_451_);
lean_dec(v_i_448_);
v_i_448_ = v___x_452_;
goto _start;
}
v___jp_454_:
{
if (v___y_455_ == 0)
{
goto v___jp_450_;
}
else
{
lean_dec(v_i_448_);
return v___y_455_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Mathlib_Tactic_higherOrderGetParam_spec__6_spec__10_spec__12___redArg___boxed(lean_object* v_keys_467_, lean_object* v_i_468_, lean_object* v_k_469_){
_start:
{
uint8_t v_res_470_; lean_object* v_r_471_; 
v_res_470_ = lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Mathlib_Tactic_higherOrderGetParam_spec__6_spec__10_spec__12___redArg(v_keys_467_, v_i_468_, v_k_469_);
lean_dec_ref(v_k_469_);
lean_dec_ref(v_keys_467_);
v_r_471_ = lean_box(v_res_470_);
return v_r_471_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Mathlib_Tactic_higherOrderGetParam_spec__6_spec__10___redArg(lean_object* v_x_472_, size_t v_x_473_, lean_object* v_x_474_){
_start:
{
if (lean_obj_tag(v_x_472_) == 0)
{
lean_object* v_es_475_; lean_object* v___x_476_; size_t v___x_477_; size_t v___x_478_; lean_object* v_j_479_; lean_object* v___x_480_; 
v_es_475_ = lean_ctor_get(v_x_472_, 0);
v___x_476_ = lean_box(2);
v___x_477_ = ((size_t)31ULL);
v___x_478_ = lean_usize_land(v_x_473_, v___x_477_);
v_j_479_ = lean_usize_to_nat(v___x_478_);
v___x_480_ = lean_array_get_borrowed(v___x_476_, v_es_475_, v_j_479_);
lean_dec(v_j_479_);
switch(lean_obj_tag(v___x_480_))
{
case 0:
{
if (lean_obj_tag(v_x_474_) == 0)
{
lean_object* v_key_481_; 
v_key_481_ = lean_ctor_get(v___x_480_, 0);
if (lean_obj_tag(v_key_481_) == 0)
{
lean_object* v_declName_482_; uint8_t v_inv_483_; lean_object* v_declName_484_; uint8_t v_inv_485_; uint8_t v___x_486_; 
v_declName_482_ = lean_ctor_get(v_x_474_, 0);
v_inv_483_ = lean_ctor_get_uint8(v_x_474_, sizeof(void*)*1 + 1);
v_declName_484_ = lean_ctor_get(v_key_481_, 0);
v_inv_485_ = lean_ctor_get_uint8(v_key_481_, sizeof(void*)*1 + 1);
v___x_486_ = lean_name_eq(v_declName_482_, v_declName_484_);
if (v___x_486_ == 0)
{
return v___x_486_;
}
else
{
if (v_inv_483_ == 0)
{
if (v_inv_485_ == 0)
{
return v___x_486_;
}
else
{
return v_inv_483_;
}
}
else
{
return v_inv_485_;
}
}
}
else
{
uint8_t v___x_487_; 
v___x_487_ = 0;
return v___x_487_;
}
}
else
{
lean_object* v_key_488_; 
v_key_488_ = lean_ctor_get(v___x_480_, 0);
if (lean_obj_tag(v_key_488_) == 0)
{
uint8_t v___x_489_; 
v___x_489_ = 0;
return v___x_489_;
}
else
{
lean_object* v___x_490_; lean_object* v___x_491_; uint8_t v___x_492_; 
v___x_490_ = l_Lean_Meta_Origin_key(v_x_474_);
v___x_491_ = l_Lean_Meta_Origin_key(v_key_488_);
v___x_492_ = lean_name_eq(v___x_490_, v___x_491_);
lean_dec(v___x_491_);
lean_dec(v___x_490_);
return v___x_492_;
}
}
}
case 1:
{
lean_object* v_node_493_; size_t v___x_494_; size_t v___x_495_; 
v_node_493_ = lean_ctor_get(v___x_480_, 0);
v___x_494_ = ((size_t)5ULL);
v___x_495_ = lean_usize_shift_right(v_x_473_, v___x_494_);
v_x_472_ = v_node_493_;
v_x_473_ = v___x_495_;
goto _start;
}
default: 
{
uint8_t v___x_497_; 
v___x_497_ = 0;
return v___x_497_;
}
}
}
else
{
lean_object* v_ks_498_; lean_object* v___x_499_; uint8_t v___x_500_; 
v_ks_498_ = lean_ctor_get(v_x_472_, 0);
v___x_499_ = lean_unsigned_to_nat(0u);
v___x_500_ = lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Mathlib_Tactic_higherOrderGetParam_spec__6_spec__10_spec__12___redArg(v_ks_498_, v___x_499_, v_x_474_);
return v___x_500_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Mathlib_Tactic_higherOrderGetParam_spec__6_spec__10___redArg___boxed(lean_object* v_x_501_, lean_object* v_x_502_, lean_object* v_x_503_){
_start:
{
size_t v_x_18091__boxed_504_; uint8_t v_res_505_; lean_object* v_r_506_; 
v_x_18091__boxed_504_ = lean_unbox_usize(v_x_502_);
lean_dec(v_x_502_);
v_res_505_ = lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Mathlib_Tactic_higherOrderGetParam_spec__6_spec__10___redArg(v_x_501_, v_x_18091__boxed_504_, v_x_503_);
lean_dec_ref(v_x_503_);
lean_dec_ref(v_x_501_);
v_r_506_ = lean_box(v_res_505_);
return v_r_506_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_contains___at___00Mathlib_Tactic_higherOrderGetParam_spec__6___redArg(lean_object* v_x_507_, lean_object* v_x_508_){
_start:
{
uint64_t v___y_510_; uint64_t v___y_514_; uint64_t v___y_518_; 
if (lean_obj_tag(v_x_508_) == 0)
{
uint8_t v_inv_521_; 
v_inv_521_ = lean_ctor_get_uint8(v_x_508_, sizeof(void*)*1 + 1);
if (v_inv_521_ == 0)
{
lean_object* v_declName_522_; 
v_declName_522_ = lean_ctor_get(v_x_508_, 0);
if (lean_obj_tag(v_declName_522_) == 0)
{
uint64_t v___x_523_; 
v___x_523_ = 1723ULL;
v___y_514_ = v___x_523_;
goto v___jp_513_;
}
else
{
uint64_t v_hash_524_; 
v_hash_524_ = lean_ctor_get_uint64(v_declName_522_, sizeof(void*)*2);
v___y_514_ = v_hash_524_;
goto v___jp_513_;
}
}
else
{
lean_object* v_declName_525_; 
v_declName_525_ = lean_ctor_get(v_x_508_, 0);
if (lean_obj_tag(v_declName_525_) == 0)
{
uint64_t v___x_526_; 
v___x_526_ = 1723ULL;
v___y_518_ = v___x_526_;
goto v___jp_517_;
}
else
{
uint64_t v_hash_527_; 
v_hash_527_ = lean_ctor_get_uint64(v_declName_525_, sizeof(void*)*2);
v___y_518_ = v_hash_527_;
goto v___jp_517_;
}
}
}
else
{
lean_object* v___x_528_; 
v___x_528_ = l_Lean_Meta_Origin_key(v_x_508_);
if (lean_obj_tag(v___x_528_) == 0)
{
uint64_t v___x_529_; 
v___x_529_ = 1723ULL;
v___y_510_ = v___x_529_;
goto v___jp_509_;
}
else
{
uint64_t v_hash_530_; 
v_hash_530_ = lean_ctor_get_uint64(v___x_528_, sizeof(void*)*2);
lean_dec(v___x_528_);
v___y_510_ = v_hash_530_;
goto v___jp_509_;
}
}
v___jp_509_:
{
size_t v___x_511_; uint8_t v___x_512_; 
v___x_511_ = lean_uint64_to_usize(v___y_510_);
v___x_512_ = lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Mathlib_Tactic_higherOrderGetParam_spec__6_spec__10___redArg(v_x_507_, v___x_511_, v_x_508_);
return v___x_512_;
}
v___jp_513_:
{
uint64_t v___x_515_; uint64_t v___x_516_; 
v___x_515_ = 13ULL;
v___x_516_ = lean_uint64_mix_hash(v___y_514_, v___x_515_);
v___y_510_ = v___x_516_;
goto v___jp_509_;
}
v___jp_517_:
{
uint64_t v___x_519_; uint64_t v___x_520_; 
v___x_519_ = 11ULL;
v___x_520_ = lean_uint64_mix_hash(v___y_518_, v___x_519_);
v___y_510_ = v___x_520_;
goto v___jp_509_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_contains___at___00Mathlib_Tactic_higherOrderGetParam_spec__6___redArg___boxed(lean_object* v_x_531_, lean_object* v_x_532_){
_start:
{
uint8_t v_res_533_; lean_object* v_r_534_; 
v_res_533_ = lp_mathlib_Lean_PersistentHashMap_contains___at___00Mathlib_Tactic_higherOrderGetParam_spec__6___redArg(v_x_531_, v_x_532_);
lean_dec_ref(v_x_532_);
lean_dec_ref(v_x_531_);
v_r_534_ = lean_box(v_res_533_);
return v_r_534_;
}
}
static lean_object* _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3_spec__12_spec__17_spec__20___redArg___closed__0(void){
_start:
{
lean_object* v___x_535_; 
v___x_535_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_535_;
}
}
static lean_object* _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3_spec__12_spec__17_spec__20___redArg___closed__1(void){
_start:
{
lean_object* v___x_536_; lean_object* v___x_537_; 
v___x_536_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3_spec__12_spec__17_spec__20___redArg___closed__0, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3_spec__12_spec__17_spec__20___redArg___closed__0_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3_spec__12_spec__17_spec__20___redArg___closed__0);
v___x_537_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_537_, 0, v___x_536_);
return v___x_537_;
}
}
static lean_object* _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3_spec__12_spec__17_spec__20___redArg___closed__2(void){
_start:
{
lean_object* v___x_538_; lean_object* v___x_539_; lean_object* v___x_540_; 
v___x_538_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3_spec__12_spec__17_spec__20___redArg___closed__1, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3_spec__12_spec__17_spec__20___redArg___closed__1_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3_spec__12_spec__17_spec__20___redArg___closed__1);
v___x_539_ = lean_unsigned_to_nat(0u);
v___x_540_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v___x_540_, 0, v___x_539_);
lean_ctor_set(v___x_540_, 1, v___x_539_);
lean_ctor_set(v___x_540_, 2, v___x_539_);
lean_ctor_set(v___x_540_, 3, v___x_539_);
lean_ctor_set(v___x_540_, 4, v___x_538_);
lean_ctor_set(v___x_540_, 5, v___x_538_);
lean_ctor_set(v___x_540_, 6, v___x_538_);
lean_ctor_set(v___x_540_, 7, v___x_538_);
lean_ctor_set(v___x_540_, 8, v___x_538_);
lean_ctor_set(v___x_540_, 9, v___x_538_);
return v___x_540_;
}
}
static lean_object* _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3_spec__12_spec__17_spec__20___redArg___closed__3(void){
_start:
{
lean_object* v___x_541_; lean_object* v___x_542_; lean_object* v___x_543_; 
v___x_541_ = lean_unsigned_to_nat(32u);
v___x_542_ = lean_mk_empty_array_with_capacity(v___x_541_);
v___x_543_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_543_, 0, v___x_542_);
return v___x_543_;
}
}
static lean_object* _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3_spec__12_spec__17_spec__20___redArg___closed__4(void){
_start:
{
size_t v___x_544_; lean_object* v___x_545_; lean_object* v___x_546_; lean_object* v___x_547_; lean_object* v___x_548_; lean_object* v___x_549_; 
v___x_544_ = ((size_t)5ULL);
v___x_545_ = lean_unsigned_to_nat(0u);
v___x_546_ = lean_unsigned_to_nat(32u);
v___x_547_ = lean_mk_empty_array_with_capacity(v___x_546_);
v___x_548_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3_spec__12_spec__17_spec__20___redArg___closed__3, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3_spec__12_spec__17_spec__20___redArg___closed__3_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3_spec__12_spec__17_spec__20___redArg___closed__3);
v___x_549_ = lean_alloc_ctor(0, 4, sizeof(size_t)*1);
lean_ctor_set(v___x_549_, 0, v___x_548_);
lean_ctor_set(v___x_549_, 1, v___x_547_);
lean_ctor_set(v___x_549_, 2, v___x_545_);
lean_ctor_set(v___x_549_, 3, v___x_545_);
lean_ctor_set_usize(v___x_549_, 4, v___x_544_);
return v___x_549_;
}
}
static lean_object* _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3_spec__12_spec__17_spec__20___redArg___closed__5(void){
_start:
{
lean_object* v___x_550_; lean_object* v___x_551_; lean_object* v___x_552_; lean_object* v___x_553_; 
v___x_550_ = lean_box(1);
v___x_551_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3_spec__12_spec__17_spec__20___redArg___closed__4, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3_spec__12_spec__17_spec__20___redArg___closed__4_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3_spec__12_spec__17_spec__20___redArg___closed__4);
v___x_552_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3_spec__12_spec__17_spec__20___redArg___closed__1, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3_spec__12_spec__17_spec__20___redArg___closed__1_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3_spec__12_spec__17_spec__20___redArg___closed__1);
v___x_553_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_553_, 0, v___x_552_);
lean_ctor_set(v___x_553_, 1, v___x_551_);
lean_ctor_set(v___x_553_, 2, v___x_550_);
return v___x_553_;
}
}
static lean_object* _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3_spec__12_spec__17_spec__20___redArg___closed__7(void){
_start:
{
lean_object* v___x_555_; lean_object* v___x_556_; 
v___x_555_ = ((lean_object*)(lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3_spec__12_spec__17_spec__20___redArg___closed__6));
v___x_556_ = l_Lean_stringToMessageData(v___x_555_);
return v___x_556_;
}
}
static lean_object* _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3_spec__12_spec__17_spec__20___redArg___closed__9(void){
_start:
{
lean_object* v___x_558_; lean_object* v___x_559_; 
v___x_558_ = ((lean_object*)(lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3_spec__12_spec__17_spec__20___redArg___closed__8));
v___x_559_ = l_Lean_stringToMessageData(v___x_558_);
return v___x_559_;
}
}
static lean_object* _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3_spec__12_spec__17_spec__20___redArg___closed__11(void){
_start:
{
lean_object* v___x_561_; lean_object* v___x_562_; 
v___x_561_ = ((lean_object*)(lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3_spec__12_spec__17_spec__20___redArg___closed__10));
v___x_562_ = l_Lean_stringToMessageData(v___x_561_);
return v___x_562_;
}
}
static lean_object* _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3_spec__12_spec__17_spec__20___redArg___closed__13(void){
_start:
{
lean_object* v___x_564_; lean_object* v___x_565_; 
v___x_564_ = ((lean_object*)(lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3_spec__12_spec__17_spec__20___redArg___closed__12));
v___x_565_ = l_Lean_stringToMessageData(v___x_564_);
return v___x_565_;
}
}
static lean_object* _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3_spec__12_spec__17_spec__20___redArg___closed__15(void){
_start:
{
lean_object* v___x_567_; lean_object* v___x_568_; 
v___x_567_ = ((lean_object*)(lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3_spec__12_spec__17_spec__20___redArg___closed__14));
v___x_568_ = l_Lean_stringToMessageData(v___x_567_);
return v___x_568_;
}
}
static lean_object* _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3_spec__12_spec__17_spec__20___redArg___closed__17(void){
_start:
{
lean_object* v___x_570_; lean_object* v___x_571_; 
v___x_570_ = ((lean_object*)(lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3_spec__12_spec__17_spec__20___redArg___closed__16));
v___x_571_ = l_Lean_stringToMessageData(v___x_570_);
return v___x_571_;
}
}
static lean_object* _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3_spec__12_spec__17_spec__20___redArg___closed__19(void){
_start:
{
lean_object* v___x_573_; lean_object* v___x_574_; 
v___x_573_ = ((lean_object*)(lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3_spec__12_spec__17_spec__20___redArg___closed__18));
v___x_574_ = l_Lean_stringToMessageData(v___x_573_);
return v___x_574_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3_spec__12_spec__17_spec__20___redArg(lean_object* v_msg_575_, lean_object* v_declHint_576_, lean_object* v___y_577_){
_start:
{
lean_object* v___x_579_; lean_object* v_env_580_; uint8_t v___x_581_; 
v___x_579_ = lean_st_ref_get(v___y_577_);
v_env_580_ = lean_ctor_get(v___x_579_, 0);
lean_inc_ref(v_env_580_);
lean_dec(v___x_579_);
v___x_581_ = l_Lean_Name_isAnonymous(v_declHint_576_);
if (v___x_581_ == 0)
{
uint8_t v_isExporting_582_; 
v_isExporting_582_ = lean_ctor_get_uint8(v_env_580_, sizeof(void*)*8);
if (v_isExporting_582_ == 0)
{
lean_object* v___x_583_; 
lean_dec_ref(v_env_580_);
lean_dec(v_declHint_576_);
v___x_583_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_583_, 0, v_msg_575_);
return v___x_583_;
}
else
{
lean_object* v___x_584_; uint8_t v___x_585_; 
lean_inc_ref(v_env_580_);
v___x_584_ = l_Lean_Environment_setExporting(v_env_580_, v___x_581_);
lean_inc(v_declHint_576_);
lean_inc_ref(v___x_584_);
v___x_585_ = l_Lean_Environment_contains(v___x_584_, v_declHint_576_, v_isExporting_582_);
if (v___x_585_ == 0)
{
lean_object* v___x_586_; 
lean_dec_ref(v___x_584_);
lean_dec_ref(v_env_580_);
lean_dec(v_declHint_576_);
v___x_586_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_586_, 0, v_msg_575_);
return v___x_586_;
}
else
{
lean_object* v___x_587_; lean_object* v___x_588_; lean_object* v___x_589_; lean_object* v___x_590_; lean_object* v___x_591_; lean_object* v_c_592_; lean_object* v___x_593_; 
v___x_587_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3_spec__12_spec__17_spec__20___redArg___closed__2, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3_spec__12_spec__17_spec__20___redArg___closed__2_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3_spec__12_spec__17_spec__20___redArg___closed__2);
v___x_588_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3_spec__12_spec__17_spec__20___redArg___closed__5, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3_spec__12_spec__17_spec__20___redArg___closed__5_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3_spec__12_spec__17_spec__20___redArg___closed__5);
v___x_589_ = l_Lean_Options_empty;
v___x_590_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_590_, 0, v___x_584_);
lean_ctor_set(v___x_590_, 1, v___x_587_);
lean_ctor_set(v___x_590_, 2, v___x_588_);
lean_ctor_set(v___x_590_, 3, v___x_589_);
lean_inc(v_declHint_576_);
v___x_591_ = l_Lean_MessageData_ofConstName(v_declHint_576_, v___x_581_);
v_c_592_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v_c_592_, 0, v___x_590_);
lean_ctor_set(v_c_592_, 1, v___x_591_);
v___x_593_ = l_Lean_Environment_getModuleIdxFor_x3f(v_env_580_, v_declHint_576_);
if (lean_obj_tag(v___x_593_) == 0)
{
lean_object* v___x_594_; lean_object* v___x_595_; lean_object* v___x_596_; lean_object* v___x_597_; lean_object* v___x_598_; lean_object* v___x_599_; lean_object* v___x_600_; 
lean_dec_ref(v_env_580_);
lean_dec(v_declHint_576_);
v___x_594_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3_spec__12_spec__17_spec__20___redArg___closed__7, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3_spec__12_spec__17_spec__20___redArg___closed__7_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3_spec__12_spec__17_spec__20___redArg___closed__7);
v___x_595_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_595_, 0, v___x_594_);
lean_ctor_set(v___x_595_, 1, v_c_592_);
v___x_596_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3_spec__12_spec__17_spec__20___redArg___closed__9, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3_spec__12_spec__17_spec__20___redArg___closed__9_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3_spec__12_spec__17_spec__20___redArg___closed__9);
v___x_597_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_597_, 0, v___x_595_);
lean_ctor_set(v___x_597_, 1, v___x_596_);
v___x_598_ = l_Lean_MessageData_note(v___x_597_);
v___x_599_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_599_, 0, v_msg_575_);
lean_ctor_set(v___x_599_, 1, v___x_598_);
v___x_600_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_600_, 0, v___x_599_);
return v___x_600_;
}
else
{
lean_object* v_val_601_; lean_object* v___x_603_; uint8_t v_isShared_604_; uint8_t v_isSharedCheck_636_; 
v_val_601_ = lean_ctor_get(v___x_593_, 0);
v_isSharedCheck_636_ = !lean_is_exclusive(v___x_593_);
if (v_isSharedCheck_636_ == 0)
{
v___x_603_ = v___x_593_;
v_isShared_604_ = v_isSharedCheck_636_;
goto v_resetjp_602_;
}
else
{
lean_inc(v_val_601_);
lean_dec(v___x_593_);
v___x_603_ = lean_box(0);
v_isShared_604_ = v_isSharedCheck_636_;
goto v_resetjp_602_;
}
v_resetjp_602_:
{
lean_object* v___x_605_; lean_object* v___x_606_; lean_object* v___x_607_; lean_object* v_mod_608_; uint8_t v___x_609_; 
v___x_605_ = lean_box(0);
v___x_606_ = l_Lean_Environment_header(v_env_580_);
lean_dec_ref(v_env_580_);
v___x_607_ = l_Lean_EnvironmentHeader_moduleNames(v___x_606_);
v_mod_608_ = lean_array_get(v___x_605_, v___x_607_, v_val_601_);
lean_dec(v_val_601_);
lean_dec_ref(v___x_607_);
v___x_609_ = l_Lean_isPrivateName(v_declHint_576_);
lean_dec(v_declHint_576_);
if (v___x_609_ == 0)
{
lean_object* v___x_610_; lean_object* v___x_611_; lean_object* v___x_612_; lean_object* v___x_613_; lean_object* v___x_614_; lean_object* v___x_615_; lean_object* v___x_616_; lean_object* v___x_617_; lean_object* v___x_618_; lean_object* v___x_619_; lean_object* v___x_621_; 
v___x_610_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3_spec__12_spec__17_spec__20___redArg___closed__11, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3_spec__12_spec__17_spec__20___redArg___closed__11_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3_spec__12_spec__17_spec__20___redArg___closed__11);
v___x_611_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_611_, 0, v___x_610_);
lean_ctor_set(v___x_611_, 1, v_c_592_);
v___x_612_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3_spec__12_spec__17_spec__20___redArg___closed__13, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3_spec__12_spec__17_spec__20___redArg___closed__13_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3_spec__12_spec__17_spec__20___redArg___closed__13);
v___x_613_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_613_, 0, v___x_611_);
lean_ctor_set(v___x_613_, 1, v___x_612_);
v___x_614_ = l_Lean_MessageData_ofName(v_mod_608_);
v___x_615_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_615_, 0, v___x_613_);
lean_ctor_set(v___x_615_, 1, v___x_614_);
v___x_616_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3_spec__12_spec__17_spec__20___redArg___closed__15, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3_spec__12_spec__17_spec__20___redArg___closed__15_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3_spec__12_spec__17_spec__20___redArg___closed__15);
v___x_617_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_617_, 0, v___x_615_);
lean_ctor_set(v___x_617_, 1, v___x_616_);
v___x_618_ = l_Lean_MessageData_note(v___x_617_);
v___x_619_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_619_, 0, v_msg_575_);
lean_ctor_set(v___x_619_, 1, v___x_618_);
if (v_isShared_604_ == 0)
{
lean_ctor_set_tag(v___x_603_, 0);
lean_ctor_set(v___x_603_, 0, v___x_619_);
v___x_621_ = v___x_603_;
goto v_reusejp_620_;
}
else
{
lean_object* v_reuseFailAlloc_622_; 
v_reuseFailAlloc_622_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_622_, 0, v___x_619_);
v___x_621_ = v_reuseFailAlloc_622_;
goto v_reusejp_620_;
}
v_reusejp_620_:
{
return v___x_621_;
}
}
else
{
lean_object* v___x_623_; lean_object* v___x_624_; lean_object* v___x_625_; lean_object* v___x_626_; lean_object* v___x_627_; lean_object* v___x_628_; lean_object* v___x_629_; lean_object* v___x_630_; lean_object* v___x_631_; lean_object* v___x_632_; lean_object* v___x_634_; 
v___x_623_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3_spec__12_spec__17_spec__20___redArg___closed__7, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3_spec__12_spec__17_spec__20___redArg___closed__7_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3_spec__12_spec__17_spec__20___redArg___closed__7);
v___x_624_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_624_, 0, v___x_623_);
lean_ctor_set(v___x_624_, 1, v_c_592_);
v___x_625_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3_spec__12_spec__17_spec__20___redArg___closed__17, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3_spec__12_spec__17_spec__20___redArg___closed__17_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3_spec__12_spec__17_spec__20___redArg___closed__17);
v___x_626_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_626_, 0, v___x_624_);
lean_ctor_set(v___x_626_, 1, v___x_625_);
v___x_627_ = l_Lean_MessageData_ofName(v_mod_608_);
v___x_628_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_628_, 0, v___x_626_);
lean_ctor_set(v___x_628_, 1, v___x_627_);
v___x_629_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3_spec__12_spec__17_spec__20___redArg___closed__19, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3_spec__12_spec__17_spec__20___redArg___closed__19_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3_spec__12_spec__17_spec__20___redArg___closed__19);
v___x_630_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_630_, 0, v___x_628_);
lean_ctor_set(v___x_630_, 1, v___x_629_);
v___x_631_ = l_Lean_MessageData_note(v___x_630_);
v___x_632_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_632_, 0, v_msg_575_);
lean_ctor_set(v___x_632_, 1, v___x_631_);
if (v_isShared_604_ == 0)
{
lean_ctor_set_tag(v___x_603_, 0);
lean_ctor_set(v___x_603_, 0, v___x_632_);
v___x_634_ = v___x_603_;
goto v_reusejp_633_;
}
else
{
lean_object* v_reuseFailAlloc_635_; 
v_reuseFailAlloc_635_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_635_, 0, v___x_632_);
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
}
else
{
lean_object* v___x_637_; 
lean_dec_ref(v_env_580_);
lean_dec(v_declHint_576_);
v___x_637_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_637_, 0, v_msg_575_);
return v___x_637_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3_spec__12_spec__17_spec__20___redArg___boxed(lean_object* v_msg_638_, lean_object* v_declHint_639_, lean_object* v___y_640_, lean_object* v___y_641_){
_start:
{
lean_object* v_res_642_; 
v_res_642_ = lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3_spec__12_spec__17_spec__20___redArg(v_msg_638_, v_declHint_639_, v___y_640_);
lean_dec(v___y_640_);
return v_res_642_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3_spec__12_spec__17(lean_object* v_msg_643_, lean_object* v_declHint_644_, lean_object* v___y_645_, lean_object* v___y_646_, lean_object* v___y_647_, lean_object* v___y_648_, lean_object* v___y_649_, lean_object* v___y_650_){
_start:
{
lean_object* v___x_652_; lean_object* v_a_653_; lean_object* v___x_655_; uint8_t v_isShared_656_; uint8_t v_isSharedCheck_662_; 
v___x_652_ = lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3_spec__12_spec__17_spec__20___redArg(v_msg_643_, v_declHint_644_, v___y_650_);
v_a_653_ = lean_ctor_get(v___x_652_, 0);
v_isSharedCheck_662_ = !lean_is_exclusive(v___x_652_);
if (v_isSharedCheck_662_ == 0)
{
v___x_655_ = v___x_652_;
v_isShared_656_ = v_isSharedCheck_662_;
goto v_resetjp_654_;
}
else
{
lean_inc(v_a_653_);
lean_dec(v___x_652_);
v___x_655_ = lean_box(0);
v_isShared_656_ = v_isSharedCheck_662_;
goto v_resetjp_654_;
}
v_resetjp_654_:
{
lean_object* v___x_657_; lean_object* v___x_658_; lean_object* v___x_660_; 
v___x_657_ = l_Lean_unknownIdentifierMessageTag;
v___x_658_ = lean_alloc_ctor(8, 2, 0);
lean_ctor_set(v___x_658_, 0, v___x_657_);
lean_ctor_set(v___x_658_, 1, v_a_653_);
if (v_isShared_656_ == 0)
{
lean_ctor_set(v___x_655_, 0, v___x_658_);
v___x_660_ = v___x_655_;
goto v_reusejp_659_;
}
else
{
lean_object* v_reuseFailAlloc_661_; 
v_reuseFailAlloc_661_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_661_, 0, v___x_658_);
v___x_660_ = v_reuseFailAlloc_661_;
goto v_reusejp_659_;
}
v_reusejp_659_:
{
return v___x_660_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3_spec__12_spec__17___boxed(lean_object* v_msg_663_, lean_object* v_declHint_664_, lean_object* v___y_665_, lean_object* v___y_666_, lean_object* v___y_667_, lean_object* v___y_668_, lean_object* v___y_669_, lean_object* v___y_670_, lean_object* v___y_671_){
_start:
{
lean_object* v_res_672_; 
v_res_672_ = lp_mathlib_Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3_spec__12_spec__17(v_msg_663_, v_declHint_664_, v___y_665_, v___y_666_, v___y_667_, v___y_668_, v___y_669_, v___y_670_);
lean_dec(v___y_670_);
lean_dec_ref(v___y_669_);
lean_dec(v___y_668_);
lean_dec_ref(v___y_667_);
lean_dec(v___y_666_);
lean_dec_ref(v___y_665_);
return v_res_672_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_higherOrderGetParam_spec__7_spec__12_spec__15(lean_object* v_opts_673_, lean_object* v_opt_674_){
_start:
{
lean_object* v_name_675_; lean_object* v_defValue_676_; lean_object* v_map_677_; lean_object* v___x_678_; 
v_name_675_ = lean_ctor_get(v_opt_674_, 0);
v_defValue_676_ = lean_ctor_get(v_opt_674_, 1);
v_map_677_ = lean_ctor_get(v_opts_673_, 0);
v___x_678_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_677_, v_name_675_);
if (lean_obj_tag(v___x_678_) == 0)
{
uint8_t v___x_679_; 
v___x_679_ = lean_unbox(v_defValue_676_);
return v___x_679_;
}
else
{
lean_object* v_val_680_; 
v_val_680_ = lean_ctor_get(v___x_678_, 0);
lean_inc(v_val_680_);
lean_dec_ref_known(v___x_678_, 1);
if (lean_obj_tag(v_val_680_) == 1)
{
uint8_t v_v_681_; 
v_v_681_ = lean_ctor_get_uint8(v_val_680_, 0);
lean_dec_ref_known(v_val_680_, 0);
return v_v_681_;
}
else
{
uint8_t v___x_682_; 
lean_dec(v_val_680_);
v___x_682_ = lean_unbox(v_defValue_676_);
return v___x_682_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_higherOrderGetParam_spec__7_spec__12_spec__15___boxed(lean_object* v_opts_683_, lean_object* v_opt_684_){
_start:
{
uint8_t v_res_685_; lean_object* v_r_686_; 
v_res_685_ = lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_higherOrderGetParam_spec__7_spec__12_spec__15(v_opts_683_, v_opt_684_);
lean_dec_ref(v_opt_684_);
lean_dec_ref(v_opts_683_);
v_r_686_ = lean_box(v_res_685_);
return v_r_686_;
}
}
static lean_object* _init_lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_higherOrderGetParam_spec__7_spec__12_spec__16___closed__0(void){
_start:
{
lean_object* v___x_687_; lean_object* v___x_688_; 
v___x_687_ = lean_box(1);
v___x_688_ = l_Lean_MessageData_ofFormat(v___x_687_);
return v___x_688_;
}
}
static lean_object* _init_lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_higherOrderGetParam_spec__7_spec__12_spec__16___closed__3(void){
_start:
{
lean_object* v___x_692_; lean_object* v___x_693_; 
v___x_692_ = ((lean_object*)(lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_higherOrderGetParam_spec__7_spec__12_spec__16___closed__2));
v___x_693_ = l_Lean_MessageData_ofFormat(v___x_692_);
return v___x_693_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_higherOrderGetParam_spec__7_spec__12_spec__16(lean_object* v_x_694_, lean_object* v_x_695_){
_start:
{
if (lean_obj_tag(v_x_695_) == 0)
{
return v_x_694_;
}
else
{
lean_object* v_head_696_; lean_object* v_tail_697_; lean_object* v___x_699_; uint8_t v_isShared_700_; uint8_t v_isSharedCheck_719_; 
v_head_696_ = lean_ctor_get(v_x_695_, 0);
v_tail_697_ = lean_ctor_get(v_x_695_, 1);
v_isSharedCheck_719_ = !lean_is_exclusive(v_x_695_);
if (v_isSharedCheck_719_ == 0)
{
v___x_699_ = v_x_695_;
v_isShared_700_ = v_isSharedCheck_719_;
goto v_resetjp_698_;
}
else
{
lean_inc(v_tail_697_);
lean_inc(v_head_696_);
lean_dec(v_x_695_);
v___x_699_ = lean_box(0);
v_isShared_700_ = v_isSharedCheck_719_;
goto v_resetjp_698_;
}
v_resetjp_698_:
{
lean_object* v_before_701_; lean_object* v___x_703_; uint8_t v_isShared_704_; uint8_t v_isSharedCheck_717_; 
v_before_701_ = lean_ctor_get(v_head_696_, 0);
v_isSharedCheck_717_ = !lean_is_exclusive(v_head_696_);
if (v_isSharedCheck_717_ == 0)
{
lean_object* v_unused_718_; 
v_unused_718_ = lean_ctor_get(v_head_696_, 1);
lean_dec(v_unused_718_);
v___x_703_ = v_head_696_;
v_isShared_704_ = v_isSharedCheck_717_;
goto v_resetjp_702_;
}
else
{
lean_inc(v_before_701_);
lean_dec(v_head_696_);
v___x_703_ = lean_box(0);
v_isShared_704_ = v_isSharedCheck_717_;
goto v_resetjp_702_;
}
v_resetjp_702_:
{
lean_object* v___x_705_; lean_object* v___x_707_; 
v___x_705_ = lean_obj_once(&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_higherOrderGetParam_spec__7_spec__12_spec__16___closed__0, &lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_higherOrderGetParam_spec__7_spec__12_spec__16___closed__0_once, _init_lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_higherOrderGetParam_spec__7_spec__12_spec__16___closed__0);
if (v_isShared_704_ == 0)
{
lean_ctor_set_tag(v___x_703_, 7);
lean_ctor_set(v___x_703_, 1, v___x_705_);
lean_ctor_set(v___x_703_, 0, v_x_694_);
v___x_707_ = v___x_703_;
goto v_reusejp_706_;
}
else
{
lean_object* v_reuseFailAlloc_716_; 
v_reuseFailAlloc_716_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_716_, 0, v_x_694_);
lean_ctor_set(v_reuseFailAlloc_716_, 1, v___x_705_);
v___x_707_ = v_reuseFailAlloc_716_;
goto v_reusejp_706_;
}
v_reusejp_706_:
{
lean_object* v___x_708_; lean_object* v___x_710_; 
v___x_708_ = lean_obj_once(&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_higherOrderGetParam_spec__7_spec__12_spec__16___closed__3, &lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_higherOrderGetParam_spec__7_spec__12_spec__16___closed__3_once, _init_lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_higherOrderGetParam_spec__7_spec__12_spec__16___closed__3);
if (v_isShared_700_ == 0)
{
lean_ctor_set_tag(v___x_699_, 7);
lean_ctor_set(v___x_699_, 1, v___x_708_);
lean_ctor_set(v___x_699_, 0, v___x_707_);
v___x_710_ = v___x_699_;
goto v_reusejp_709_;
}
else
{
lean_object* v_reuseFailAlloc_715_; 
v_reuseFailAlloc_715_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_715_, 0, v___x_707_);
lean_ctor_set(v_reuseFailAlloc_715_, 1, v___x_708_);
v___x_710_ = v_reuseFailAlloc_715_;
goto v_reusejp_709_;
}
v_reusejp_709_:
{
lean_object* v___x_711_; lean_object* v___x_712_; lean_object* v___x_713_; 
v___x_711_ = l_Lean_MessageData_ofSyntax(v_before_701_);
v___x_712_ = l_Lean_indentD(v___x_711_);
v___x_713_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_713_, 0, v___x_710_);
lean_ctor_set(v___x_713_, 1, v___x_712_);
v_x_694_ = v___x_713_;
v_x_695_ = v_tail_697_;
goto _start;
}
}
}
}
}
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_higherOrderGetParam_spec__7_spec__12___redArg___closed__2(void){
_start:
{
lean_object* v___x_723_; lean_object* v___x_724_; 
v___x_723_ = ((lean_object*)(lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_higherOrderGetParam_spec__7_spec__12___redArg___closed__1));
v___x_724_ = l_Lean_MessageData_ofFormat(v___x_723_);
return v___x_724_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_higherOrderGetParam_spec__7_spec__12___redArg(lean_object* v_msgData_725_, lean_object* v_macroStack_726_, lean_object* v___y_727_){
_start:
{
lean_object* v_options_729_; lean_object* v___x_730_; uint8_t v___x_731_; 
v_options_729_ = lean_ctor_get(v___y_727_, 2);
v___x_730_ = l_Lean_Elab_pp_macroStack;
v___x_731_ = lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_higherOrderGetParam_spec__7_spec__12_spec__15(v_options_729_, v___x_730_);
if (v___x_731_ == 0)
{
lean_object* v___x_732_; 
lean_dec(v_macroStack_726_);
v___x_732_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_732_, 0, v_msgData_725_);
return v___x_732_;
}
else
{
if (lean_obj_tag(v_macroStack_726_) == 0)
{
lean_object* v___x_733_; 
v___x_733_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_733_, 0, v_msgData_725_);
return v___x_733_;
}
else
{
lean_object* v_head_734_; lean_object* v_after_735_; lean_object* v___x_737_; uint8_t v_isShared_738_; uint8_t v_isSharedCheck_750_; 
v_head_734_ = lean_ctor_get(v_macroStack_726_, 0);
lean_inc(v_head_734_);
v_after_735_ = lean_ctor_get(v_head_734_, 1);
v_isSharedCheck_750_ = !lean_is_exclusive(v_head_734_);
if (v_isSharedCheck_750_ == 0)
{
lean_object* v_unused_751_; 
v_unused_751_ = lean_ctor_get(v_head_734_, 0);
lean_dec(v_unused_751_);
v___x_737_ = v_head_734_;
v_isShared_738_ = v_isSharedCheck_750_;
goto v_resetjp_736_;
}
else
{
lean_inc(v_after_735_);
lean_dec(v_head_734_);
v___x_737_ = lean_box(0);
v_isShared_738_ = v_isSharedCheck_750_;
goto v_resetjp_736_;
}
v_resetjp_736_:
{
lean_object* v___x_739_; lean_object* v___x_741_; 
v___x_739_ = lean_obj_once(&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_higherOrderGetParam_spec__7_spec__12_spec__16___closed__0, &lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_higherOrderGetParam_spec__7_spec__12_spec__16___closed__0_once, _init_lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_higherOrderGetParam_spec__7_spec__12_spec__16___closed__0);
if (v_isShared_738_ == 0)
{
lean_ctor_set_tag(v___x_737_, 7);
lean_ctor_set(v___x_737_, 1, v___x_739_);
lean_ctor_set(v___x_737_, 0, v_msgData_725_);
v___x_741_ = v___x_737_;
goto v_reusejp_740_;
}
else
{
lean_object* v_reuseFailAlloc_749_; 
v_reuseFailAlloc_749_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_749_, 0, v_msgData_725_);
lean_ctor_set(v_reuseFailAlloc_749_, 1, v___x_739_);
v___x_741_ = v_reuseFailAlloc_749_;
goto v_reusejp_740_;
}
v_reusejp_740_:
{
lean_object* v___x_742_; lean_object* v___x_743_; lean_object* v___x_744_; lean_object* v___x_745_; lean_object* v_msgData_746_; lean_object* v___x_747_; lean_object* v___x_748_; 
v___x_742_ = lean_obj_once(&lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_higherOrderGetParam_spec__7_spec__12___redArg___closed__2, &lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_higherOrderGetParam_spec__7_spec__12___redArg___closed__2_once, _init_lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_higherOrderGetParam_spec__7_spec__12___redArg___closed__2);
v___x_743_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_743_, 0, v___x_741_);
lean_ctor_set(v___x_743_, 1, v___x_742_);
v___x_744_ = l_Lean_MessageData_ofSyntax(v_after_735_);
v___x_745_ = l_Lean_indentD(v___x_744_);
v_msgData_746_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_msgData_746_, 0, v___x_743_);
lean_ctor_set(v_msgData_746_, 1, v___x_745_);
v___x_747_ = lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_higherOrderGetParam_spec__7_spec__12_spec__16(v_msgData_746_, v_macroStack_726_);
v___x_748_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_748_, 0, v___x_747_);
return v___x_748_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_higherOrderGetParam_spec__7_spec__12___redArg___boxed(lean_object* v_msgData_752_, lean_object* v_macroStack_753_, lean_object* v___y_754_, lean_object* v___y_755_){
_start:
{
lean_object* v_res_756_; 
v_res_756_ = lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_higherOrderGetParam_spec__7_spec__12___redArg(v_msgData_752_, v_macroStack_753_, v___y_754_);
lean_dec_ref(v___y_754_);
return v_res_756_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_higherOrderGetParam_spec__7___redArg(lean_object* v_msg_757_, lean_object* v___y_758_, lean_object* v___y_759_, lean_object* v___y_760_, lean_object* v___y_761_, lean_object* v___y_762_, lean_object* v___y_763_){
_start:
{
lean_object* v_ref_765_; lean_object* v___x_766_; lean_object* v_a_767_; lean_object* v_macroStack_768_; lean_object* v___x_769_; lean_object* v___x_770_; lean_object* v_a_771_; lean_object* v___x_773_; uint8_t v_isShared_774_; uint8_t v_isSharedCheck_779_; 
v_ref_765_ = lean_ctor_get(v___y_762_, 5);
v___x_766_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_mkComp_spec__0_spec__0(v_msg_757_, v___y_760_, v___y_761_, v___y_762_, v___y_763_);
v_a_767_ = lean_ctor_get(v___x_766_, 0);
lean_inc(v_a_767_);
lean_dec_ref(v___x_766_);
v_macroStack_768_ = lean_ctor_get(v___y_758_, 1);
v___x_769_ = l_Lean_Elab_getBetterRef(v_ref_765_, v_macroStack_768_);
lean_inc(v_macroStack_768_);
v___x_770_ = lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_higherOrderGetParam_spec__7_spec__12___redArg(v_a_767_, v_macroStack_768_, v___y_762_);
v_a_771_ = lean_ctor_get(v___x_770_, 0);
v_isSharedCheck_779_ = !lean_is_exclusive(v___x_770_);
if (v_isSharedCheck_779_ == 0)
{
v___x_773_ = v___x_770_;
v_isShared_774_ = v_isSharedCheck_779_;
goto v_resetjp_772_;
}
else
{
lean_inc(v_a_771_);
lean_dec(v___x_770_);
v___x_773_ = lean_box(0);
v_isShared_774_ = v_isSharedCheck_779_;
goto v_resetjp_772_;
}
v_resetjp_772_:
{
lean_object* v___x_775_; lean_object* v___x_777_; 
v___x_775_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_775_, 0, v___x_769_);
lean_ctor_set(v___x_775_, 1, v_a_771_);
if (v_isShared_774_ == 0)
{
lean_ctor_set_tag(v___x_773_, 1);
lean_ctor_set(v___x_773_, 0, v___x_775_);
v___x_777_ = v___x_773_;
goto v_reusejp_776_;
}
else
{
lean_object* v_reuseFailAlloc_778_; 
v_reuseFailAlloc_778_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_778_, 0, v___x_775_);
v___x_777_ = v_reuseFailAlloc_778_;
goto v_reusejp_776_;
}
v_reusejp_776_:
{
return v___x_777_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_higherOrderGetParam_spec__7___redArg___boxed(lean_object* v_msg_780_, lean_object* v___y_781_, lean_object* v___y_782_, lean_object* v___y_783_, lean_object* v___y_784_, lean_object* v___y_785_, lean_object* v___y_786_, lean_object* v___y_787_){
_start:
{
lean_object* v_res_788_; 
v_res_788_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_higherOrderGetParam_spec__7___redArg(v_msg_780_, v___y_781_, v___y_782_, v___y_783_, v___y_784_, v___y_785_, v___y_786_);
lean_dec(v___y_786_);
lean_dec_ref(v___y_785_);
lean_dec(v___y_784_);
lean_dec_ref(v___y_783_);
lean_dec(v___y_782_);
lean_dec_ref(v___y_781_);
return v_res_788_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3_spec__12_spec__18___redArg(lean_object* v_ref_789_, lean_object* v_msg_790_, lean_object* v___y_791_, lean_object* v___y_792_, lean_object* v___y_793_, lean_object* v___y_794_, lean_object* v___y_795_, lean_object* v___y_796_){
_start:
{
lean_object* v_fileName_798_; lean_object* v_fileMap_799_; lean_object* v_options_800_; lean_object* v_currRecDepth_801_; lean_object* v_maxRecDepth_802_; lean_object* v_ref_803_; lean_object* v_currNamespace_804_; lean_object* v_openDecls_805_; lean_object* v_initHeartbeats_806_; lean_object* v_maxHeartbeats_807_; lean_object* v_quotContext_808_; lean_object* v_currMacroScope_809_; uint8_t v_diag_810_; lean_object* v_cancelTk_x3f_811_; uint8_t v_suppressElabErrors_812_; lean_object* v_inheritedTraceOptions_813_; lean_object* v_ref_814_; lean_object* v___x_815_; lean_object* v___x_816_; 
v_fileName_798_ = lean_ctor_get(v___y_795_, 0);
v_fileMap_799_ = lean_ctor_get(v___y_795_, 1);
v_options_800_ = lean_ctor_get(v___y_795_, 2);
v_currRecDepth_801_ = lean_ctor_get(v___y_795_, 3);
v_maxRecDepth_802_ = lean_ctor_get(v___y_795_, 4);
v_ref_803_ = lean_ctor_get(v___y_795_, 5);
v_currNamespace_804_ = lean_ctor_get(v___y_795_, 6);
v_openDecls_805_ = lean_ctor_get(v___y_795_, 7);
v_initHeartbeats_806_ = lean_ctor_get(v___y_795_, 8);
v_maxHeartbeats_807_ = lean_ctor_get(v___y_795_, 9);
v_quotContext_808_ = lean_ctor_get(v___y_795_, 10);
v_currMacroScope_809_ = lean_ctor_get(v___y_795_, 11);
v_diag_810_ = lean_ctor_get_uint8(v___y_795_, sizeof(void*)*14);
v_cancelTk_x3f_811_ = lean_ctor_get(v___y_795_, 12);
v_suppressElabErrors_812_ = lean_ctor_get_uint8(v___y_795_, sizeof(void*)*14 + 1);
v_inheritedTraceOptions_813_ = lean_ctor_get(v___y_795_, 13);
v_ref_814_ = l_Lean_replaceRef(v_ref_789_, v_ref_803_);
lean_inc_ref(v_inheritedTraceOptions_813_);
lean_inc(v_cancelTk_x3f_811_);
lean_inc(v_currMacroScope_809_);
lean_inc(v_quotContext_808_);
lean_inc(v_maxHeartbeats_807_);
lean_inc(v_initHeartbeats_806_);
lean_inc(v_openDecls_805_);
lean_inc(v_currNamespace_804_);
lean_inc(v_maxRecDepth_802_);
lean_inc(v_currRecDepth_801_);
lean_inc_ref(v_options_800_);
lean_inc_ref(v_fileMap_799_);
lean_inc_ref(v_fileName_798_);
v___x_815_ = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(v___x_815_, 0, v_fileName_798_);
lean_ctor_set(v___x_815_, 1, v_fileMap_799_);
lean_ctor_set(v___x_815_, 2, v_options_800_);
lean_ctor_set(v___x_815_, 3, v_currRecDepth_801_);
lean_ctor_set(v___x_815_, 4, v_maxRecDepth_802_);
lean_ctor_set(v___x_815_, 5, v_ref_814_);
lean_ctor_set(v___x_815_, 6, v_currNamespace_804_);
lean_ctor_set(v___x_815_, 7, v_openDecls_805_);
lean_ctor_set(v___x_815_, 8, v_initHeartbeats_806_);
lean_ctor_set(v___x_815_, 9, v_maxHeartbeats_807_);
lean_ctor_set(v___x_815_, 10, v_quotContext_808_);
lean_ctor_set(v___x_815_, 11, v_currMacroScope_809_);
lean_ctor_set(v___x_815_, 12, v_cancelTk_x3f_811_);
lean_ctor_set(v___x_815_, 13, v_inheritedTraceOptions_813_);
lean_ctor_set_uint8(v___x_815_, sizeof(void*)*14, v_diag_810_);
lean_ctor_set_uint8(v___x_815_, sizeof(void*)*14 + 1, v_suppressElabErrors_812_);
v___x_816_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_higherOrderGetParam_spec__7___redArg(v_msg_790_, v___y_791_, v___y_792_, v___y_793_, v___y_794_, v___x_815_, v___y_796_);
lean_dec_ref_known(v___x_815_, 14);
return v___x_816_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3_spec__12_spec__18___redArg___boxed(lean_object* v_ref_817_, lean_object* v_msg_818_, lean_object* v___y_819_, lean_object* v___y_820_, lean_object* v___y_821_, lean_object* v___y_822_, lean_object* v___y_823_, lean_object* v___y_824_, lean_object* v___y_825_){
_start:
{
lean_object* v_res_826_; 
v_res_826_ = lp_mathlib_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3_spec__12_spec__18___redArg(v_ref_817_, v_msg_818_, v___y_819_, v___y_820_, v___y_821_, v___y_822_, v___y_823_, v___y_824_);
lean_dec(v___y_824_);
lean_dec_ref(v___y_823_);
lean_dec(v___y_822_);
lean_dec_ref(v___y_821_);
lean_dec(v___y_820_);
lean_dec_ref(v___y_819_);
lean_dec(v_ref_817_);
return v_res_826_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3_spec__12___redArg(lean_object* v_ref_827_, lean_object* v_msg_828_, lean_object* v_declHint_829_, lean_object* v___y_830_, lean_object* v___y_831_, lean_object* v___y_832_, lean_object* v___y_833_, lean_object* v___y_834_, lean_object* v___y_835_){
_start:
{
lean_object* v___x_837_; lean_object* v_a_838_; lean_object* v___x_839_; 
v___x_837_ = lp_mathlib_Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3_spec__12_spec__17(v_msg_828_, v_declHint_829_, v___y_830_, v___y_831_, v___y_832_, v___y_833_, v___y_834_, v___y_835_);
v_a_838_ = lean_ctor_get(v___x_837_, 0);
lean_inc(v_a_838_);
lean_dec_ref(v___x_837_);
v___x_839_ = lp_mathlib_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3_spec__12_spec__18___redArg(v_ref_827_, v_a_838_, v___y_830_, v___y_831_, v___y_832_, v___y_833_, v___y_834_, v___y_835_);
return v___x_839_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3_spec__12___redArg___boxed(lean_object* v_ref_840_, lean_object* v_msg_841_, lean_object* v_declHint_842_, lean_object* v___y_843_, lean_object* v___y_844_, lean_object* v___y_845_, lean_object* v___y_846_, lean_object* v___y_847_, lean_object* v___y_848_, lean_object* v___y_849_){
_start:
{
lean_object* v_res_850_; 
v_res_850_ = lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3_spec__12___redArg(v_ref_840_, v_msg_841_, v_declHint_842_, v___y_843_, v___y_844_, v___y_845_, v___y_846_, v___y_847_, v___y_848_);
lean_dec(v___y_848_);
lean_dec_ref(v___y_847_);
lean_dec(v___y_846_);
lean_dec_ref(v___y_845_);
lean_dec(v___y_844_);
lean_dec_ref(v___y_843_);
lean_dec(v_ref_840_);
return v_res_850_;
}
}
static lean_object* _init_lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3___redArg___closed__1(void){
_start:
{
lean_object* v___x_852_; lean_object* v___x_853_; 
v___x_852_ = ((lean_object*)(lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3___redArg___closed__0));
v___x_853_ = l_Lean_stringToMessageData(v___x_852_);
return v___x_853_;
}
}
static lean_object* _init_lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3___redArg___closed__3(void){
_start:
{
lean_object* v___x_855_; lean_object* v___x_856_; 
v___x_855_ = ((lean_object*)(lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3___redArg___closed__2));
v___x_856_ = l_Lean_stringToMessageData(v___x_855_);
return v___x_856_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3___redArg(lean_object* v_ref_857_, lean_object* v_constName_858_, lean_object* v___y_859_, lean_object* v___y_860_, lean_object* v___y_861_, lean_object* v___y_862_, lean_object* v___y_863_, lean_object* v___y_864_){
_start:
{
lean_object* v___x_866_; uint8_t v___x_867_; lean_object* v___x_868_; lean_object* v___x_869_; lean_object* v___x_870_; lean_object* v___x_871_; lean_object* v___x_872_; 
v___x_866_ = lean_obj_once(&lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3___redArg___closed__1, &lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3___redArg___closed__1_once, _init_lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3___redArg___closed__1);
v___x_867_ = 0;
lean_inc(v_constName_858_);
v___x_868_ = l_Lean_MessageData_ofConstName(v_constName_858_, v___x_867_);
v___x_869_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_869_, 0, v___x_866_);
lean_ctor_set(v___x_869_, 1, v___x_868_);
v___x_870_ = lean_obj_once(&lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3___redArg___closed__3, &lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3___redArg___closed__3_once, _init_lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3___redArg___closed__3);
v___x_871_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_871_, 0, v___x_869_);
lean_ctor_set(v___x_871_, 1, v___x_870_);
v___x_872_ = lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3_spec__12___redArg(v_ref_857_, v___x_871_, v_constName_858_, v___y_859_, v___y_860_, v___y_861_, v___y_862_, v___y_863_, v___y_864_);
return v___x_872_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3___redArg___boxed(lean_object* v_ref_873_, lean_object* v_constName_874_, lean_object* v___y_875_, lean_object* v___y_876_, lean_object* v___y_877_, lean_object* v___y_878_, lean_object* v___y_879_, lean_object* v___y_880_, lean_object* v___y_881_){
_start:
{
lean_object* v_res_882_; 
v_res_882_ = lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3___redArg(v_ref_873_, v_constName_874_, v___y_875_, v___y_876_, v___y_877_, v___y_878_, v___y_879_, v___y_880_);
lean_dec(v___y_880_);
lean_dec_ref(v___y_879_);
lean_dec(v___y_878_);
lean_dec_ref(v___y_877_);
lean_dec(v___y_876_);
lean_dec_ref(v___y_875_);
lean_dec(v_ref_873_);
return v_res_882_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1___redArg(lean_object* v_constName_883_, lean_object* v___y_884_, lean_object* v___y_885_, lean_object* v___y_886_, lean_object* v___y_887_, lean_object* v___y_888_, lean_object* v___y_889_){
_start:
{
lean_object* v_ref_891_; lean_object* v___x_892_; 
v_ref_891_ = lean_ctor_get(v___y_888_, 5);
v___x_892_ = lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3___redArg(v_ref_891_, v_constName_883_, v___y_884_, v___y_885_, v___y_886_, v___y_887_, v___y_888_, v___y_889_);
return v___x_892_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1___redArg___boxed(lean_object* v_constName_893_, lean_object* v___y_894_, lean_object* v___y_895_, lean_object* v___y_896_, lean_object* v___y_897_, lean_object* v___y_898_, lean_object* v___y_899_, lean_object* v___y_900_){
_start:
{
lean_object* v_res_901_; 
v_res_901_ = lp_mathlib_Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1___redArg(v_constName_893_, v___y_894_, v___y_895_, v___y_896_, v___y_897_, v___y_898_, v___y_899_);
lean_dec(v___y_899_);
lean_dec_ref(v___y_898_);
lean_dec(v___y_897_);
lean_dec_ref(v___y_896_);
lean_dec(v___y_895_);
lean_dec_ref(v___y_894_);
return v_res_901_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_higherOrderGetParam_spec__5_spec__8(lean_object* v_constName_902_, lean_object* v___y_903_, lean_object* v___y_904_, lean_object* v___y_905_, lean_object* v___y_906_, lean_object* v___y_907_, lean_object* v___y_908_){
_start:
{
lean_object* v___x_910_; lean_object* v_env_911_; uint8_t v___x_912_; lean_object* v___x_913_; 
v___x_910_ = lean_st_ref_get(v___y_908_);
v_env_911_ = lean_ctor_get(v___x_910_, 0);
lean_inc_ref(v_env_911_);
lean_dec(v___x_910_);
v___x_912_ = 0;
lean_inc(v_constName_902_);
v___x_913_ = l_Lean_Environment_findConstVal_x3f(v_env_911_, v_constName_902_, v___x_912_);
if (lean_obj_tag(v___x_913_) == 0)
{
lean_object* v___x_914_; 
v___x_914_ = lp_mathlib_Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1___redArg(v_constName_902_, v___y_903_, v___y_904_, v___y_905_, v___y_906_, v___y_907_, v___y_908_);
return v___x_914_;
}
else
{
lean_object* v_val_915_; lean_object* v___x_917_; uint8_t v_isShared_918_; uint8_t v_isSharedCheck_922_; 
lean_dec(v_constName_902_);
v_val_915_ = lean_ctor_get(v___x_913_, 0);
v_isSharedCheck_922_ = !lean_is_exclusive(v___x_913_);
if (v_isSharedCheck_922_ == 0)
{
v___x_917_ = v___x_913_;
v_isShared_918_ = v_isSharedCheck_922_;
goto v_resetjp_916_;
}
else
{
lean_inc(v_val_915_);
lean_dec(v___x_913_);
v___x_917_ = lean_box(0);
v_isShared_918_ = v_isSharedCheck_922_;
goto v_resetjp_916_;
}
v_resetjp_916_:
{
lean_object* v___x_920_; 
if (v_isShared_918_ == 0)
{
lean_ctor_set_tag(v___x_917_, 0);
v___x_920_ = v___x_917_;
goto v_reusejp_919_;
}
else
{
lean_object* v_reuseFailAlloc_921_; 
v_reuseFailAlloc_921_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_921_, 0, v_val_915_);
v___x_920_ = v_reuseFailAlloc_921_;
goto v_reusejp_919_;
}
v_reusejp_919_:
{
return v___x_920_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_higherOrderGetParam_spec__5_spec__8___boxed(lean_object* v_constName_923_, lean_object* v___y_924_, lean_object* v___y_925_, lean_object* v___y_926_, lean_object* v___y_927_, lean_object* v___y_928_, lean_object* v___y_929_, lean_object* v___y_930_){
_start:
{
lean_object* v_res_931_; 
v_res_931_ = lp_mathlib_Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_higherOrderGetParam_spec__5_spec__8(v_constName_923_, v___y_924_, v___y_925_, v___y_926_, v___y_927_, v___y_928_, v___y_929_);
lean_dec(v___y_929_);
lean_dec_ref(v___y_928_);
lean_dec(v___y_927_);
lean_dec_ref(v___y_926_);
lean_dec(v___y_925_);
lean_dec_ref(v___y_924_);
return v_res_931_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_higherOrderGetParam_spec__2(lean_object* v_a_932_, lean_object* v_a_933_){
_start:
{
if (lean_obj_tag(v_a_932_) == 0)
{
lean_object* v___x_934_; 
v___x_934_ = l_List_reverse___redArg(v_a_933_);
return v___x_934_;
}
else
{
lean_object* v_head_935_; lean_object* v_tail_936_; lean_object* v___x_938_; uint8_t v_isShared_939_; uint8_t v_isSharedCheck_945_; 
v_head_935_ = lean_ctor_get(v_a_932_, 0);
v_tail_936_ = lean_ctor_get(v_a_932_, 1);
v_isSharedCheck_945_ = !lean_is_exclusive(v_a_932_);
if (v_isSharedCheck_945_ == 0)
{
v___x_938_ = v_a_932_;
v_isShared_939_ = v_isSharedCheck_945_;
goto v_resetjp_937_;
}
else
{
lean_inc(v_tail_936_);
lean_inc(v_head_935_);
lean_dec(v_a_932_);
v___x_938_ = lean_box(0);
v_isShared_939_ = v_isSharedCheck_945_;
goto v_resetjp_937_;
}
v_resetjp_937_:
{
lean_object* v___x_940_; lean_object* v___x_942_; 
v___x_940_ = l_Lean_mkLevelParam(v_head_935_);
if (v_isShared_939_ == 0)
{
lean_ctor_set(v___x_938_, 1, v_a_933_);
lean_ctor_set(v___x_938_, 0, v___x_940_);
v___x_942_ = v___x_938_;
goto v_reusejp_941_;
}
else
{
lean_object* v_reuseFailAlloc_944_; 
v_reuseFailAlloc_944_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_944_, 0, v___x_940_);
lean_ctor_set(v_reuseFailAlloc_944_, 1, v_a_933_);
v___x_942_ = v_reuseFailAlloc_944_;
goto v_reusejp_941_;
}
v_reusejp_941_:
{
v_a_932_ = v_tail_936_;
v_a_933_ = v___x_942_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_higherOrderGetParam_spec__5(lean_object* v_constName_946_, lean_object* v___y_947_, lean_object* v___y_948_, lean_object* v___y_949_, lean_object* v___y_950_, lean_object* v___y_951_, lean_object* v___y_952_){
_start:
{
lean_object* v___x_954_; 
lean_inc(v_constName_946_);
v___x_954_ = lp_mathlib_Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_higherOrderGetParam_spec__5_spec__8(v_constName_946_, v___y_947_, v___y_948_, v___y_949_, v___y_950_, v___y_951_, v___y_952_);
if (lean_obj_tag(v___x_954_) == 0)
{
lean_object* v_a_955_; lean_object* v___x_957_; uint8_t v_isShared_958_; uint8_t v_isSharedCheck_966_; 
v_a_955_ = lean_ctor_get(v___x_954_, 0);
v_isSharedCheck_966_ = !lean_is_exclusive(v___x_954_);
if (v_isSharedCheck_966_ == 0)
{
v___x_957_ = v___x_954_;
v_isShared_958_ = v_isSharedCheck_966_;
goto v_resetjp_956_;
}
else
{
lean_inc(v_a_955_);
lean_dec(v___x_954_);
v___x_957_ = lean_box(0);
v_isShared_958_ = v_isSharedCheck_966_;
goto v_resetjp_956_;
}
v_resetjp_956_:
{
lean_object* v_levelParams_959_; lean_object* v___x_960_; lean_object* v___x_961_; lean_object* v___x_962_; lean_object* v___x_964_; 
v_levelParams_959_ = lean_ctor_get(v_a_955_, 1);
lean_inc(v_levelParams_959_);
lean_dec(v_a_955_);
v___x_960_ = lean_box(0);
v___x_961_ = lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_higherOrderGetParam_spec__2(v_levelParams_959_, v___x_960_);
v___x_962_ = l_Lean_mkConst(v_constName_946_, v___x_961_);
if (v_isShared_958_ == 0)
{
lean_ctor_set(v___x_957_, 0, v___x_962_);
v___x_964_ = v___x_957_;
goto v_reusejp_963_;
}
else
{
lean_object* v_reuseFailAlloc_965_; 
v_reuseFailAlloc_965_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_965_, 0, v___x_962_);
v___x_964_ = v_reuseFailAlloc_965_;
goto v_reusejp_963_;
}
v_reusejp_963_:
{
return v___x_964_;
}
}
}
else
{
lean_object* v_a_967_; lean_object* v___x_969_; uint8_t v_isShared_970_; uint8_t v_isSharedCheck_974_; 
lean_dec(v_constName_946_);
v_a_967_ = lean_ctor_get(v___x_954_, 0);
v_isSharedCheck_974_ = !lean_is_exclusive(v___x_954_);
if (v_isSharedCheck_974_ == 0)
{
v___x_969_ = v___x_954_;
v_isShared_970_ = v_isSharedCheck_974_;
goto v_resetjp_968_;
}
else
{
lean_inc(v_a_967_);
lean_dec(v___x_954_);
v___x_969_ = lean_box(0);
v_isShared_970_ = v_isSharedCheck_974_;
goto v_resetjp_968_;
}
v_resetjp_968_:
{
lean_object* v___x_972_; 
if (v_isShared_970_ == 0)
{
v___x_972_ = v___x_969_;
goto v_reusejp_971_;
}
else
{
lean_object* v_reuseFailAlloc_973_; 
v_reuseFailAlloc_973_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_973_, 0, v_a_967_);
v___x_972_ = v_reuseFailAlloc_973_;
goto v_reusejp_971_;
}
v_reusejp_971_:
{
return v___x_972_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_higherOrderGetParam_spec__5___boxed(lean_object* v_constName_975_, lean_object* v___y_976_, lean_object* v___y_977_, lean_object* v___y_978_, lean_object* v___y_979_, lean_object* v___y_980_, lean_object* v___y_981_, lean_object* v___y_982_){
_start:
{
lean_object* v_res_983_; 
v_res_983_ = lp_mathlib_Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_higherOrderGetParam_spec__5(v_constName_975_, v___y_976_, v___y_977_, v___y_978_, v___y_979_, v___y_980_, v___y_981_);
lean_dec(v___y_981_);
lean_dec_ref(v___y_980_);
lean_dec(v___y_979_);
lean_dec_ref(v___y_978_);
lean_dec(v___y_977_);
lean_dec_ref(v___y_976_);
return v_res_983_;
}
}
static lean_object* _init_lp_mathlib_Lean_addDeclarationRanges___at___00Lean_Elab_addDeclarationRangesFromSyntax___at___00Mathlib_Tactic_higherOrderGetParam_spec__4_spec__6___redArg___closed__0(void){
_start:
{
lean_object* v___x_984_; 
v___x_984_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_984_;
}
}
static lean_object* _init_lp_mathlib_Lean_addDeclarationRanges___at___00Lean_Elab_addDeclarationRangesFromSyntax___at___00Mathlib_Tactic_higherOrderGetParam_spec__4_spec__6___redArg___closed__1(void){
_start:
{
lean_object* v___x_985_; lean_object* v___x_986_; 
v___x_985_ = lean_obj_once(&lp_mathlib_Lean_addDeclarationRanges___at___00Lean_Elab_addDeclarationRangesFromSyntax___at___00Mathlib_Tactic_higherOrderGetParam_spec__4_spec__6___redArg___closed__0, &lp_mathlib_Lean_addDeclarationRanges___at___00Lean_Elab_addDeclarationRangesFromSyntax___at___00Mathlib_Tactic_higherOrderGetParam_spec__4_spec__6___redArg___closed__0_once, _init_lp_mathlib_Lean_addDeclarationRanges___at___00Lean_Elab_addDeclarationRangesFromSyntax___at___00Mathlib_Tactic_higherOrderGetParam_spec__4_spec__6___redArg___closed__0);
v___x_986_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_986_, 0, v___x_985_);
return v___x_986_;
}
}
static lean_object* _init_lp_mathlib_Lean_addDeclarationRanges___at___00Lean_Elab_addDeclarationRangesFromSyntax___at___00Mathlib_Tactic_higherOrderGetParam_spec__4_spec__6___redArg___closed__2(void){
_start:
{
lean_object* v___x_987_; lean_object* v___x_988_; 
v___x_987_ = lean_obj_once(&lp_mathlib_Lean_addDeclarationRanges___at___00Lean_Elab_addDeclarationRangesFromSyntax___at___00Mathlib_Tactic_higherOrderGetParam_spec__4_spec__6___redArg___closed__1, &lp_mathlib_Lean_addDeclarationRanges___at___00Lean_Elab_addDeclarationRangesFromSyntax___at___00Mathlib_Tactic_higherOrderGetParam_spec__4_spec__6___redArg___closed__1_once, _init_lp_mathlib_Lean_addDeclarationRanges___at___00Lean_Elab_addDeclarationRangesFromSyntax___at___00Mathlib_Tactic_higherOrderGetParam_spec__4_spec__6___redArg___closed__1);
v___x_988_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_988_, 0, v___x_987_);
lean_ctor_set(v___x_988_, 1, v___x_987_);
return v___x_988_;
}
}
static lean_object* _init_lp_mathlib_Lean_addDeclarationRanges___at___00Lean_Elab_addDeclarationRangesFromSyntax___at___00Mathlib_Tactic_higherOrderGetParam_spec__4_spec__6___redArg___closed__3(void){
_start:
{
lean_object* v___x_989_; lean_object* v___x_990_; 
v___x_989_ = lean_obj_once(&lp_mathlib_Lean_addDeclarationRanges___at___00Lean_Elab_addDeclarationRangesFromSyntax___at___00Mathlib_Tactic_higherOrderGetParam_spec__4_spec__6___redArg___closed__1, &lp_mathlib_Lean_addDeclarationRanges___at___00Lean_Elab_addDeclarationRangesFromSyntax___at___00Mathlib_Tactic_higherOrderGetParam_spec__4_spec__6___redArg___closed__1_once, _init_lp_mathlib_Lean_addDeclarationRanges___at___00Lean_Elab_addDeclarationRangesFromSyntax___at___00Mathlib_Tactic_higherOrderGetParam_spec__4_spec__6___redArg___closed__1);
v___x_990_ = lean_alloc_ctor(0, 6, 0);
lean_ctor_set(v___x_990_, 0, v___x_989_);
lean_ctor_set(v___x_990_, 1, v___x_989_);
lean_ctor_set(v___x_990_, 2, v___x_989_);
lean_ctor_set(v___x_990_, 3, v___x_989_);
lean_ctor_set(v___x_990_, 4, v___x_989_);
lean_ctor_set(v___x_990_, 5, v___x_989_);
return v___x_990_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addDeclarationRanges___at___00Lean_Elab_addDeclarationRangesFromSyntax___at___00Mathlib_Tactic_higherOrderGetParam_spec__4_spec__6___redArg(lean_object* v_declName_991_, lean_object* v_declRanges_992_, lean_object* v___y_993_, lean_object* v___y_994_){
_start:
{
uint8_t v___x_996_; 
v___x_996_ = l_Lean_Name_isAnonymous(v_declName_991_);
if (v___x_996_ == 0)
{
lean_object* v___x_997_; lean_object* v_env_998_; lean_object* v_nextMacroScope_999_; lean_object* v_ngen_1000_; lean_object* v_auxDeclNGen_1001_; lean_object* v_traceState_1002_; lean_object* v_messages_1003_; lean_object* v_infoState_1004_; lean_object* v_snapshotTasks_1005_; lean_object* v___x_1007_; uint8_t v_isShared_1008_; uint8_t v_isSharedCheck_1033_; 
v___x_997_ = lean_st_ref_take(v___y_994_);
v_env_998_ = lean_ctor_get(v___x_997_, 0);
v_nextMacroScope_999_ = lean_ctor_get(v___x_997_, 1);
v_ngen_1000_ = lean_ctor_get(v___x_997_, 2);
v_auxDeclNGen_1001_ = lean_ctor_get(v___x_997_, 3);
v_traceState_1002_ = lean_ctor_get(v___x_997_, 4);
v_messages_1003_ = lean_ctor_get(v___x_997_, 6);
v_infoState_1004_ = lean_ctor_get(v___x_997_, 7);
v_snapshotTasks_1005_ = lean_ctor_get(v___x_997_, 8);
v_isSharedCheck_1033_ = !lean_is_exclusive(v___x_997_);
if (v_isSharedCheck_1033_ == 0)
{
lean_object* v_unused_1034_; 
v_unused_1034_ = lean_ctor_get(v___x_997_, 5);
lean_dec(v_unused_1034_);
v___x_1007_ = v___x_997_;
v_isShared_1008_ = v_isSharedCheck_1033_;
goto v_resetjp_1006_;
}
else
{
lean_inc(v_snapshotTasks_1005_);
lean_inc(v_infoState_1004_);
lean_inc(v_messages_1003_);
lean_inc(v_traceState_1002_);
lean_inc(v_auxDeclNGen_1001_);
lean_inc(v_ngen_1000_);
lean_inc(v_nextMacroScope_999_);
lean_inc(v_env_998_);
lean_dec(v___x_997_);
v___x_1007_ = lean_box(0);
v_isShared_1008_ = v_isSharedCheck_1033_;
goto v_resetjp_1006_;
}
v_resetjp_1006_:
{
lean_object* v___x_1009_; lean_object* v___x_1010_; lean_object* v___x_1011_; lean_object* v___x_1013_; 
v___x_1009_ = l_Lean_declRangeExt;
v___x_1010_ = l_Lean_MapDeclarationExtension_insert___redArg(v___x_1009_, v_env_998_, v_declName_991_, v_declRanges_992_);
v___x_1011_ = lean_obj_once(&lp_mathlib_Lean_addDeclarationRanges___at___00Lean_Elab_addDeclarationRangesFromSyntax___at___00Mathlib_Tactic_higherOrderGetParam_spec__4_spec__6___redArg___closed__2, &lp_mathlib_Lean_addDeclarationRanges___at___00Lean_Elab_addDeclarationRangesFromSyntax___at___00Mathlib_Tactic_higherOrderGetParam_spec__4_spec__6___redArg___closed__2_once, _init_lp_mathlib_Lean_addDeclarationRanges___at___00Lean_Elab_addDeclarationRangesFromSyntax___at___00Mathlib_Tactic_higherOrderGetParam_spec__4_spec__6___redArg___closed__2);
if (v_isShared_1008_ == 0)
{
lean_ctor_set(v___x_1007_, 5, v___x_1011_);
lean_ctor_set(v___x_1007_, 0, v___x_1010_);
v___x_1013_ = v___x_1007_;
goto v_reusejp_1012_;
}
else
{
lean_object* v_reuseFailAlloc_1032_; 
v_reuseFailAlloc_1032_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_1032_, 0, v___x_1010_);
lean_ctor_set(v_reuseFailAlloc_1032_, 1, v_nextMacroScope_999_);
lean_ctor_set(v_reuseFailAlloc_1032_, 2, v_ngen_1000_);
lean_ctor_set(v_reuseFailAlloc_1032_, 3, v_auxDeclNGen_1001_);
lean_ctor_set(v_reuseFailAlloc_1032_, 4, v_traceState_1002_);
lean_ctor_set(v_reuseFailAlloc_1032_, 5, v___x_1011_);
lean_ctor_set(v_reuseFailAlloc_1032_, 6, v_messages_1003_);
lean_ctor_set(v_reuseFailAlloc_1032_, 7, v_infoState_1004_);
lean_ctor_set(v_reuseFailAlloc_1032_, 8, v_snapshotTasks_1005_);
v___x_1013_ = v_reuseFailAlloc_1032_;
goto v_reusejp_1012_;
}
v_reusejp_1012_:
{
lean_object* v___x_1014_; lean_object* v___x_1015_; lean_object* v_mctx_1016_; lean_object* v_zetaDeltaFVarIds_1017_; lean_object* v_postponed_1018_; lean_object* v_diag_1019_; lean_object* v___x_1021_; uint8_t v_isShared_1022_; uint8_t v_isSharedCheck_1030_; 
v___x_1014_ = lean_st_ref_set(v___y_994_, v___x_1013_);
v___x_1015_ = lean_st_ref_take(v___y_993_);
v_mctx_1016_ = lean_ctor_get(v___x_1015_, 0);
v_zetaDeltaFVarIds_1017_ = lean_ctor_get(v___x_1015_, 2);
v_postponed_1018_ = lean_ctor_get(v___x_1015_, 3);
v_diag_1019_ = lean_ctor_get(v___x_1015_, 4);
v_isSharedCheck_1030_ = !lean_is_exclusive(v___x_1015_);
if (v_isSharedCheck_1030_ == 0)
{
lean_object* v_unused_1031_; 
v_unused_1031_ = lean_ctor_get(v___x_1015_, 1);
lean_dec(v_unused_1031_);
v___x_1021_ = v___x_1015_;
v_isShared_1022_ = v_isSharedCheck_1030_;
goto v_resetjp_1020_;
}
else
{
lean_inc(v_diag_1019_);
lean_inc(v_postponed_1018_);
lean_inc(v_zetaDeltaFVarIds_1017_);
lean_inc(v_mctx_1016_);
lean_dec(v___x_1015_);
v___x_1021_ = lean_box(0);
v_isShared_1022_ = v_isSharedCheck_1030_;
goto v_resetjp_1020_;
}
v_resetjp_1020_:
{
lean_object* v___x_1023_; lean_object* v___x_1025_; 
v___x_1023_ = lean_obj_once(&lp_mathlib_Lean_addDeclarationRanges___at___00Lean_Elab_addDeclarationRangesFromSyntax___at___00Mathlib_Tactic_higherOrderGetParam_spec__4_spec__6___redArg___closed__3, &lp_mathlib_Lean_addDeclarationRanges___at___00Lean_Elab_addDeclarationRangesFromSyntax___at___00Mathlib_Tactic_higherOrderGetParam_spec__4_spec__6___redArg___closed__3_once, _init_lp_mathlib_Lean_addDeclarationRanges___at___00Lean_Elab_addDeclarationRangesFromSyntax___at___00Mathlib_Tactic_higherOrderGetParam_spec__4_spec__6___redArg___closed__3);
if (v_isShared_1022_ == 0)
{
lean_ctor_set(v___x_1021_, 1, v___x_1023_);
v___x_1025_ = v___x_1021_;
goto v_reusejp_1024_;
}
else
{
lean_object* v_reuseFailAlloc_1029_; 
v_reuseFailAlloc_1029_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_1029_, 0, v_mctx_1016_);
lean_ctor_set(v_reuseFailAlloc_1029_, 1, v___x_1023_);
lean_ctor_set(v_reuseFailAlloc_1029_, 2, v_zetaDeltaFVarIds_1017_);
lean_ctor_set(v_reuseFailAlloc_1029_, 3, v_postponed_1018_);
lean_ctor_set(v_reuseFailAlloc_1029_, 4, v_diag_1019_);
v___x_1025_ = v_reuseFailAlloc_1029_;
goto v_reusejp_1024_;
}
v_reusejp_1024_:
{
lean_object* v___x_1026_; lean_object* v___x_1027_; lean_object* v___x_1028_; 
v___x_1026_ = lean_st_ref_set(v___y_993_, v___x_1025_);
v___x_1027_ = lean_box(0);
v___x_1028_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1028_, 0, v___x_1027_);
return v___x_1028_;
}
}
}
}
}
else
{
lean_object* v___x_1035_; lean_object* v___x_1036_; 
lean_dec_ref(v_declRanges_992_);
lean_dec(v_declName_991_);
v___x_1035_ = lean_box(0);
v___x_1036_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1036_, 0, v___x_1035_);
return v___x_1036_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addDeclarationRanges___at___00Lean_Elab_addDeclarationRangesFromSyntax___at___00Mathlib_Tactic_higherOrderGetParam_spec__4_spec__6___redArg___boxed(lean_object* v_declName_1037_, lean_object* v_declRanges_1038_, lean_object* v___y_1039_, lean_object* v___y_1040_, lean_object* v___y_1041_){
_start:
{
lean_object* v_res_1042_; 
v_res_1042_ = lp_mathlib_Lean_addDeclarationRanges___at___00Lean_Elab_addDeclarationRangesFromSyntax___at___00Mathlib_Tactic_higherOrderGetParam_spec__4_spec__6___redArg(v_declName_1037_, v_declRanges_1038_, v___y_1039_, v___y_1040_);
lean_dec(v___y_1040_);
lean_dec(v___y_1039_);
return v_res_1042_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_getDeclarationRange_x3f___at___00Lean_Elab_addDeclarationRangesFromSyntax___at___00Mathlib_Tactic_higherOrderGetParam_spec__4_spec__5___redArg(lean_object* v_stx_1043_, lean_object* v___y_1044_){
_start:
{
uint8_t v___x_1046_; lean_object* v___x_1047_; 
v___x_1046_ = 0;
v___x_1047_ = l_Lean_Syntax_getRange_x3f(v_stx_1043_, v___x_1046_);
if (lean_obj_tag(v___x_1047_) == 1)
{
lean_object* v_val_1048_; lean_object* v___x_1050_; uint8_t v_isShared_1051_; uint8_t v_isSharedCheck_1060_; 
v_val_1048_ = lean_ctor_get(v___x_1047_, 0);
v_isSharedCheck_1060_ = !lean_is_exclusive(v___x_1047_);
if (v_isSharedCheck_1060_ == 0)
{
v___x_1050_ = v___x_1047_;
v_isShared_1051_ = v_isSharedCheck_1060_;
goto v_resetjp_1049_;
}
else
{
lean_inc(v_val_1048_);
lean_dec(v___x_1047_);
v___x_1050_ = lean_box(0);
v_isShared_1051_ = v_isSharedCheck_1060_;
goto v_resetjp_1049_;
}
v_resetjp_1049_:
{
lean_object* v_fileMap_1052_; lean_object* v_start_1053_; lean_object* v_stop_1054_; lean_object* v___x_1055_; lean_object* v___x_1057_; 
v_fileMap_1052_ = lean_ctor_get(v___y_1044_, 1);
v_start_1053_ = lean_ctor_get(v_val_1048_, 0);
lean_inc(v_start_1053_);
v_stop_1054_ = lean_ctor_get(v_val_1048_, 1);
lean_inc(v_stop_1054_);
lean_dec(v_val_1048_);
lean_inc_ref(v_fileMap_1052_);
v___x_1055_ = l_Lean_DeclarationRange_ofStringPositions(v_fileMap_1052_, v_start_1053_, v_stop_1054_);
lean_dec(v_stop_1054_);
lean_dec(v_start_1053_);
if (v_isShared_1051_ == 0)
{
lean_ctor_set(v___x_1050_, 0, v___x_1055_);
v___x_1057_ = v___x_1050_;
goto v_reusejp_1056_;
}
else
{
lean_object* v_reuseFailAlloc_1059_; 
v_reuseFailAlloc_1059_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1059_, 0, v___x_1055_);
v___x_1057_ = v_reuseFailAlloc_1059_;
goto v_reusejp_1056_;
}
v_reusejp_1056_:
{
lean_object* v___x_1058_; 
v___x_1058_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1058_, 0, v___x_1057_);
return v___x_1058_;
}
}
}
else
{
lean_object* v___x_1061_; lean_object* v___x_1062_; 
lean_dec(v___x_1047_);
v___x_1061_ = lean_box(0);
v___x_1062_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1062_, 0, v___x_1061_);
return v___x_1062_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_getDeclarationRange_x3f___at___00Lean_Elab_addDeclarationRangesFromSyntax___at___00Mathlib_Tactic_higherOrderGetParam_spec__4_spec__5___redArg___boxed(lean_object* v_stx_1063_, lean_object* v___y_1064_, lean_object* v___y_1065_){
_start:
{
lean_object* v_res_1066_; 
v_res_1066_ = lp_mathlib_Lean_Elab_getDeclarationRange_x3f___at___00Lean_Elab_addDeclarationRangesFromSyntax___at___00Mathlib_Tactic_higherOrderGetParam_spec__4_spec__5___redArg(v_stx_1063_, v___y_1064_);
lean_dec_ref(v___y_1064_);
lean_dec(v_stx_1063_);
return v_res_1066_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addDeclarationRangesFromSyntax___at___00Mathlib_Tactic_higherOrderGetParam_spec__4(lean_object* v_declName_1067_, lean_object* v_rangeStx_1068_, lean_object* v_selectionRangeStx_1069_, lean_object* v___y_1070_, lean_object* v___y_1071_, lean_object* v___y_1072_, lean_object* v___y_1073_, lean_object* v___y_1074_, lean_object* v___y_1075_){
_start:
{
lean_object* v___x_1077_; lean_object* v_a_1078_; lean_object* v___x_1080_; uint8_t v_isShared_1081_; uint8_t v_isSharedCheck_1094_; 
v___x_1077_ = lp_mathlib_Lean_Elab_getDeclarationRange_x3f___at___00Lean_Elab_addDeclarationRangesFromSyntax___at___00Mathlib_Tactic_higherOrderGetParam_spec__4_spec__5___redArg(v_rangeStx_1068_, v___y_1074_);
v_a_1078_ = lean_ctor_get(v___x_1077_, 0);
v_isSharedCheck_1094_ = !lean_is_exclusive(v___x_1077_);
if (v_isSharedCheck_1094_ == 0)
{
v___x_1080_ = v___x_1077_;
v_isShared_1081_ = v_isSharedCheck_1094_;
goto v_resetjp_1079_;
}
else
{
lean_inc(v_a_1078_);
lean_dec(v___x_1077_);
v___x_1080_ = lean_box(0);
v_isShared_1081_ = v_isSharedCheck_1094_;
goto v_resetjp_1079_;
}
v_resetjp_1079_:
{
if (lean_obj_tag(v_a_1078_) == 1)
{
lean_object* v_val_1082_; lean_object* v___x_1083_; lean_object* v_a_1084_; lean_object* v_a_1086_; 
lean_del_object(v___x_1080_);
v_val_1082_ = lean_ctor_get(v_a_1078_, 0);
lean_inc(v_val_1082_);
lean_dec_ref_known(v_a_1078_, 1);
v___x_1083_ = lp_mathlib_Lean_Elab_getDeclarationRange_x3f___at___00Lean_Elab_addDeclarationRangesFromSyntax___at___00Mathlib_Tactic_higherOrderGetParam_spec__4_spec__5___redArg(v_selectionRangeStx_1069_, v___y_1074_);
v_a_1084_ = lean_ctor_get(v___x_1083_, 0);
lean_inc(v_a_1084_);
lean_dec_ref(v___x_1083_);
if (lean_obj_tag(v_a_1084_) == 0)
{
lean_inc(v_val_1082_);
v_a_1086_ = v_val_1082_;
goto v___jp_1085_;
}
else
{
lean_object* v_val_1089_; 
v_val_1089_ = lean_ctor_get(v_a_1084_, 0);
lean_inc(v_val_1089_);
lean_dec_ref_known(v_a_1084_, 1);
v_a_1086_ = v_val_1089_;
goto v___jp_1085_;
}
v___jp_1085_:
{
lean_object* v___x_1087_; lean_object* v___x_1088_; 
v___x_1087_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1087_, 0, v_val_1082_);
lean_ctor_set(v___x_1087_, 1, v_a_1086_);
v___x_1088_ = lp_mathlib_Lean_addDeclarationRanges___at___00Lean_Elab_addDeclarationRangesFromSyntax___at___00Mathlib_Tactic_higherOrderGetParam_spec__4_spec__6___redArg(v_declName_1067_, v___x_1087_, v___y_1073_, v___y_1075_);
return v___x_1088_;
}
}
else
{
lean_object* v___x_1090_; lean_object* v___x_1092_; 
lean_dec(v_a_1078_);
lean_dec(v_declName_1067_);
v___x_1090_ = lean_box(0);
if (v_isShared_1081_ == 0)
{
lean_ctor_set(v___x_1080_, 0, v___x_1090_);
v___x_1092_ = v___x_1080_;
goto v_reusejp_1091_;
}
else
{
lean_object* v_reuseFailAlloc_1093_; 
v_reuseFailAlloc_1093_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1093_, 0, v___x_1090_);
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
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addDeclarationRangesFromSyntax___at___00Mathlib_Tactic_higherOrderGetParam_spec__4___boxed(lean_object* v_declName_1095_, lean_object* v_rangeStx_1096_, lean_object* v_selectionRangeStx_1097_, lean_object* v___y_1098_, lean_object* v___y_1099_, lean_object* v___y_1100_, lean_object* v___y_1101_, lean_object* v___y_1102_, lean_object* v___y_1103_, lean_object* v___y_1104_){
_start:
{
lean_object* v_res_1105_; 
v_res_1105_ = lp_mathlib_Lean_Elab_addDeclarationRangesFromSyntax___at___00Mathlib_Tactic_higherOrderGetParam_spec__4(v_declName_1095_, v_rangeStx_1096_, v_selectionRangeStx_1097_, v___y_1098_, v___y_1099_, v___y_1100_, v___y_1101_, v___y_1102_, v___y_1103_);
lean_dec(v___y_1103_);
lean_dec_ref(v___y_1102_);
lean_dec(v___y_1101_);
lean_dec_ref(v___y_1100_);
lean_dec(v___y_1099_);
lean_dec_ref(v___y_1098_);
lean_dec(v_selectionRangeStx_1097_);
lean_dec(v_rangeStx_1096_);
return v_res_1105_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forM___at___00Mathlib_Tactic_higherOrderGetParam_spec__8___redArg(lean_object* v_as_1106_, lean_object* v___y_1107_, lean_object* v___y_1108_, lean_object* v___y_1109_, lean_object* v___y_1110_){
_start:
{
if (lean_obj_tag(v_as_1106_) == 0)
{
lean_object* v___x_1112_; lean_object* v___x_1113_; 
v___x_1112_ = lean_box(0);
v___x_1113_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1113_, 0, v___x_1112_);
return v___x_1113_;
}
else
{
lean_object* v_head_1114_; lean_object* v_tail_1115_; lean_object* v___x_1116_; 
v_head_1114_ = lean_ctor_get(v_as_1106_, 0);
lean_inc(v_head_1114_);
v_tail_1115_ = lean_ctor_get(v_as_1106_, 1);
lean_inc(v_tail_1115_);
lean_dec_ref_known(v_as_1106_, 2);
v___x_1116_ = l_Lean_MVarId_assumption(v_head_1114_, v___y_1107_, v___y_1108_, v___y_1109_, v___y_1110_);
if (lean_obj_tag(v___x_1116_) == 0)
{
lean_dec_ref_known(v___x_1116_, 1);
v_as_1106_ = v_tail_1115_;
goto _start;
}
else
{
lean_dec(v_tail_1115_);
return v___x_1116_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forM___at___00Mathlib_Tactic_higherOrderGetParam_spec__8___redArg___boxed(lean_object* v_as_1118_, lean_object* v___y_1119_, lean_object* v___y_1120_, lean_object* v___y_1121_, lean_object* v___y_1122_, lean_object* v___y_1123_){
_start:
{
lean_object* v_res_1124_; 
v_res_1124_ = lp_mathlib_List_forM___at___00Mathlib_Tactic_higherOrderGetParam_spec__8___redArg(v_as_1118_, v___y_1119_, v___y_1120_, v___y_1121_, v___y_1122_);
lean_dec(v___y_1122_);
lean_dec_ref(v___y_1121_);
lean_dec(v___y_1120_);
lean_dec_ref(v___y_1119_);
return v_res_1124_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1(lean_object* v_constName_1125_, lean_object* v___y_1126_, lean_object* v___y_1127_, lean_object* v___y_1128_, lean_object* v___y_1129_, lean_object* v___y_1130_, lean_object* v___y_1131_){
_start:
{
lean_object* v___x_1133_; lean_object* v_env_1134_; uint8_t v___x_1135_; lean_object* v___x_1136_; 
v___x_1133_ = lean_st_ref_get(v___y_1131_);
v_env_1134_ = lean_ctor_get(v___x_1133_, 0);
lean_inc_ref(v_env_1134_);
lean_dec(v___x_1133_);
v___x_1135_ = 0;
lean_inc(v_constName_1125_);
v___x_1136_ = l_Lean_Environment_find_x3f(v_env_1134_, v_constName_1125_, v___x_1135_);
if (lean_obj_tag(v___x_1136_) == 0)
{
lean_object* v___x_1137_; 
v___x_1137_ = lp_mathlib_Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1___redArg(v_constName_1125_, v___y_1126_, v___y_1127_, v___y_1128_, v___y_1129_, v___y_1130_, v___y_1131_);
return v___x_1137_;
}
else
{
lean_object* v_val_1138_; lean_object* v___x_1140_; uint8_t v_isShared_1141_; uint8_t v_isSharedCheck_1145_; 
lean_dec(v_constName_1125_);
v_val_1138_ = lean_ctor_get(v___x_1136_, 0);
v_isSharedCheck_1145_ = !lean_is_exclusive(v___x_1136_);
if (v_isSharedCheck_1145_ == 0)
{
v___x_1140_ = v___x_1136_;
v_isShared_1141_ = v_isSharedCheck_1145_;
goto v_resetjp_1139_;
}
else
{
lean_inc(v_val_1138_);
lean_dec(v___x_1136_);
v___x_1140_ = lean_box(0);
v_isShared_1141_ = v_isSharedCheck_1145_;
goto v_resetjp_1139_;
}
v_resetjp_1139_:
{
lean_object* v___x_1143_; 
if (v_isShared_1141_ == 0)
{
lean_ctor_set_tag(v___x_1140_, 0);
v___x_1143_ = v___x_1140_;
goto v_reusejp_1142_;
}
else
{
lean_object* v_reuseFailAlloc_1144_; 
v_reuseFailAlloc_1144_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1144_, 0, v_val_1138_);
v___x_1143_ = v_reuseFailAlloc_1144_;
goto v_reusejp_1142_;
}
v_reusejp_1142_:
{
return v___x_1143_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1___boxed(lean_object* v_constName_1146_, lean_object* v___y_1147_, lean_object* v___y_1148_, lean_object* v___y_1149_, lean_object* v___y_1150_, lean_object* v___y_1151_, lean_object* v___y_1152_, lean_object* v___y_1153_){
_start:
{
lean_object* v_res_1154_; 
v_res_1154_ = lp_mathlib_Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1(v_constName_1146_, v___y_1147_, v___y_1148_, v___y_1149_, v___y_1150_, v___y_1151_, v___y_1152_);
lean_dec(v___y_1152_);
lean_dec_ref(v___y_1151_);
lean_dec(v___y_1150_);
lean_dec_ref(v___y_1149_);
lean_dec(v___y_1148_);
lean_dec_ref(v___y_1147_);
return v_res_1154_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_higherOrderGetParam___lam__1(lean_object* v_thm_1161_, uint8_t v___x_1162_, lean_object* v___y_1163_, lean_object* v___y_1164_, lean_object* v___y_1165_, lean_object* v___y_1166_, lean_object* v___y_1167_, lean_object* v___y_1168_, lean_object* v___y_1169_, lean_object* v___y_1170_){
_start:
{
lean_object* v___x_1172_; 
lean_inc(v_thm_1161_);
v___x_1172_ = lp_mathlib_Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1(v_thm_1161_, v___y_1165_, v___y_1166_, v___y_1167_, v___y_1168_, v___y_1169_, v___y_1170_);
if (lean_obj_tag(v___x_1172_) == 0)
{
lean_object* v_a_1173_; lean_object* v___x_1174_; lean_object* v___x_1175_; lean_object* v___x_1176_; lean_object* v___x_1177_; lean_object* v___x_1178_; 
v_a_1173_ = lean_ctor_get(v___x_1172_, 0);
lean_inc(v_a_1173_);
lean_dec_ref_known(v___x_1172_, 1);
v___x_1174_ = l_Lean_ConstantInfo_levelParams(v_a_1173_);
lean_dec(v_a_1173_);
v___x_1175_ = lean_box(0);
lean_inc(v___x_1174_);
v___x_1176_ = lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_higherOrderGetParam_spec__2(v___x_1174_, v___x_1175_);
lean_inc(v_thm_1161_);
v___x_1177_ = l_Lean_Expr_const___override(v_thm_1161_, v___x_1176_);
lean_inc(v___y_1170_);
lean_inc_ref(v___y_1169_);
lean_inc(v___y_1168_);
lean_inc_ref(v___y_1167_);
v___x_1178_ = lean_infer_type(v___x_1177_, v___y_1167_, v___y_1168_, v___y_1169_, v___y_1170_);
if (lean_obj_tag(v___x_1178_) == 0)
{
lean_object* v_a_1179_; lean_object* v___x_1180_; lean_object* v_a_1181_; lean_object* v___x_1183_; uint8_t v_isShared_1184_; uint8_t v_isSharedCheck_1443_; 
v_a_1179_ = lean_ctor_get(v___x_1178_, 0);
lean_inc(v_a_1179_);
lean_dec_ref_known(v___x_1178_, 1);
v___x_1180_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_higherOrderGetParam_spec__3___redArg(v_a_1179_, v___y_1168_);
v_a_1181_ = lean_ctor_get(v___x_1180_, 0);
v_isSharedCheck_1443_ = !lean_is_exclusive(v___x_1180_);
if (v_isSharedCheck_1443_ == 0)
{
v___x_1183_ = v___x_1180_;
v_isShared_1184_ = v_isSharedCheck_1443_;
goto v_resetjp_1182_;
}
else
{
lean_inc(v_a_1181_);
lean_dec(v___x_1180_);
v___x_1183_ = lean_box(0);
v_isShared_1184_ = v_isSharedCheck_1443_;
goto v_resetjp_1182_;
}
v_resetjp_1182_:
{
lean_object* v___x_1185_; 
v___x_1185_ = lp_mathlib_Mathlib_Tactic_mkHigherOrderType(v_a_1181_, v___y_1167_, v___y_1168_, v___y_1169_, v___y_1170_);
if (lean_obj_tag(v___x_1185_) == 0)
{
lean_object* v_a_1186_; lean_object* v___x_1188_; 
v_a_1186_ = lean_ctor_get(v___x_1185_, 0);
lean_inc_n(v_a_1186_, 2);
lean_dec_ref_known(v___x_1185_, 1);
if (v_isShared_1184_ == 0)
{
lean_ctor_set_tag(v___x_1183_, 1);
lean_ctor_set(v___x_1183_, 0, v_a_1186_);
v___x_1188_ = v___x_1183_;
goto v_reusejp_1187_;
}
else
{
lean_object* v_reuseFailAlloc_1434_; 
v_reuseFailAlloc_1434_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1434_, 0, v_a_1186_);
v___x_1188_ = v_reuseFailAlloc_1434_;
goto v_reusejp_1187_;
}
v_reusejp_1187_:
{
uint8_t v___x_1189_; lean_object* v___x_1190_; lean_object* v___x_1191_; 
v___x_1189_ = 0;
v___x_1190_ = lean_box(0);
v___x_1191_ = l_Lean_Meta_mkFreshExprMVar(v___x_1188_, v___x_1189_, v___x_1190_, v___y_1167_, v___y_1168_, v___y_1169_, v___y_1170_);
if (lean_obj_tag(v___x_1191_) == 0)
{
lean_object* v_a_1192_; lean_object* v___x_1193_; lean_object* v___x_1194_; 
v_a_1192_ = lean_ctor_get(v___x_1191_, 0);
lean_inc(v_a_1192_);
lean_dec_ref_known(v___x_1191_, 1);
v___x_1193_ = l_Lean_Expr_mvarId_x21(v_a_1192_);
v___x_1194_ = l_Lean_MVarId_intros(v___x_1193_, v___y_1167_, v___y_1168_, v___y_1169_, v___y_1170_);
if (lean_obj_tag(v___x_1194_) == 0)
{
lean_object* v_a_1195_; lean_object* v_snd_1196_; lean_object* v___x_1198_; uint8_t v_isShared_1199_; uint8_t v_isSharedCheck_1416_; 
v_a_1195_ = lean_ctor_get(v___x_1194_, 0);
lean_inc(v_a_1195_);
lean_dec_ref_known(v___x_1194_, 1);
v_snd_1196_ = lean_ctor_get(v_a_1195_, 1);
v_isSharedCheck_1416_ = !lean_is_exclusive(v_a_1195_);
if (v_isSharedCheck_1416_ == 0)
{
lean_object* v_unused_1417_; 
v_unused_1417_ = lean_ctor_get(v_a_1195_, 0);
lean_dec(v_unused_1417_);
v___x_1198_ = v_a_1195_;
v_isShared_1199_ = v_isSharedCheck_1416_;
goto v_resetjp_1197_;
}
else
{
lean_inc(v_snd_1196_);
lean_dec(v_a_1195_);
v___x_1198_ = lean_box(0);
v_isShared_1199_ = v_isSharedCheck_1416_;
goto v_resetjp_1197_;
}
v_resetjp_1197_:
{
lean_object* v___x_1200_; lean_object* v___x_1201_; 
v___x_1200_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_higherOrderGetParam___lam__1___closed__1));
v___x_1201_ = l_Lean_Elab_Term_mkConst(v___x_1200_, v___x_1175_, v___y_1165_, v___y_1166_, v___y_1167_, v___y_1168_, v___y_1169_, v___y_1170_);
if (lean_obj_tag(v___x_1201_) == 0)
{
lean_object* v_a_1202_; uint8_t v___x_1203_; uint8_t v___x_1204_; lean_object* v___y_1206_; lean_object* v___y_1207_; lean_object* v___y_1208_; lean_object* v___y_1209_; lean_object* v___y_1210_; lean_object* v___y_1211_; lean_object* v_prf_1261_; lean_object* v___y_1262_; lean_object* v___y_1263_; lean_object* v___y_1264_; lean_object* v___y_1265_; lean_object* v___y_1266_; lean_object* v___y_1267_; lean_object* v___x_1335_; lean_object* v___x_1336_; lean_object* v___x_1337_; 
v_a_1202_ = lean_ctor_get(v___x_1201_, 0);
lean_inc(v_a_1202_);
lean_dec_ref_known(v___x_1201_, 1);
v___x_1203_ = 0;
v___x_1204_ = 0;
v___x_1335_ = lean_alloc_ctor(0, 0, 4);
lean_ctor_set_uint8(v___x_1335_, 0, v___x_1203_);
lean_ctor_set_uint8(v___x_1335_, 1, v___x_1162_);
lean_ctor_set_uint8(v___x_1335_, 2, v___x_1204_);
lean_ctor_set_uint8(v___x_1335_, 3, v___x_1162_);
v___x_1336_ = lean_box(0);
lean_inc_ref(v___x_1335_);
v___x_1337_ = l_Lean_MVarId_apply(v_snd_1196_, v_a_1202_, v___x_1335_, v___x_1336_, v___y_1167_, v___y_1168_, v___y_1169_, v___y_1170_);
if (lean_obj_tag(v___x_1337_) == 0)
{
lean_object* v_a_1338_; lean_object* v___y_1340_; lean_object* v___y_1341_; lean_object* v___y_1342_; lean_object* v___y_1343_; lean_object* v___y_1344_; lean_object* v___y_1345_; 
v_a_1338_ = lean_ctor_get(v___x_1337_, 0);
lean_inc(v_a_1338_);
lean_dec_ref_known(v___x_1337_, 1);
if (lean_obj_tag(v_a_1338_) == 1)
{
lean_object* v_tail_1356_; 
v_tail_1356_ = lean_ctor_get(v_a_1338_, 1);
if (lean_obj_tag(v_tail_1356_) == 0)
{
lean_object* v_head_1357_; lean_object* v___x_1358_; 
v_head_1357_ = lean_ctor_get(v_a_1338_, 0);
lean_inc(v_head_1357_);
lean_dec_ref_known(v_a_1338_, 2);
v___x_1358_ = l_Lean_Meta_intro1Core(v_head_1357_, v___x_1204_, v___y_1167_, v___y_1168_, v___y_1169_, v___y_1170_);
if (lean_obj_tag(v___x_1358_) == 0)
{
lean_object* v_a_1359_; lean_object* v_snd_1360_; lean_object* v___x_1361_; 
v_a_1359_ = lean_ctor_get(v___x_1358_, 0);
lean_inc(v_a_1359_);
lean_dec_ref_known(v___x_1358_, 1);
v_snd_1360_ = lean_ctor_get(v_a_1359_, 1);
lean_inc(v_snd_1360_);
lean_dec(v_a_1359_);
lean_inc(v_thm_1161_);
v___x_1361_ = l_Lean_Elab_Term_mkConst(v_thm_1161_, v___x_1175_, v___y_1165_, v___y_1166_, v___y_1167_, v___y_1168_, v___y_1169_, v___y_1170_);
if (lean_obj_tag(v___x_1361_) == 0)
{
lean_object* v_a_1362_; lean_object* v___x_1363_; 
v_a_1362_ = lean_ctor_get(v___x_1361_, 0);
lean_inc(v_a_1362_);
lean_dec_ref_known(v___x_1361_, 1);
v___x_1363_ = l_Lean_MVarId_apply(v_snd_1360_, v_a_1362_, v___x_1335_, v___x_1336_, v___y_1167_, v___y_1168_, v___y_1169_, v___y_1170_);
if (lean_obj_tag(v___x_1363_) == 0)
{
lean_object* v_a_1364_; lean_object* v___x_1365_; 
v_a_1364_ = lean_ctor_get(v___x_1363_, 0);
lean_inc(v_a_1364_);
lean_dec_ref_known(v___x_1363_, 1);
v___x_1365_ = lp_mathlib_List_forM___at___00Mathlib_Tactic_higherOrderGetParam_spec__8___redArg(v_a_1364_, v___y_1167_, v___y_1168_, v___y_1169_, v___y_1170_);
if (lean_obj_tag(v___x_1365_) == 0)
{
lean_object* v___x_1366_; lean_object* v_a_1367_; 
lean_dec_ref_known(v___x_1365_, 1);
v___x_1366_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_higherOrderGetParam_spec__3___redArg(v_a_1192_, v___y_1168_);
v_a_1367_ = lean_ctor_get(v___x_1366_, 0);
lean_inc(v_a_1367_);
lean_dec_ref(v___x_1366_);
v_prf_1261_ = v_a_1367_;
v___y_1262_ = v___y_1165_;
v___y_1263_ = v___y_1166_;
v___y_1264_ = v___y_1167_;
v___y_1265_ = v___y_1168_;
v___y_1266_ = v___y_1169_;
v___y_1267_ = v___y_1170_;
goto v___jp_1260_;
}
else
{
lean_object* v_a_1368_; lean_object* v___x_1370_; uint8_t v_isShared_1371_; uint8_t v_isSharedCheck_1375_; 
lean_del_object(v___x_1198_);
lean_dec(v_a_1192_);
lean_dec(v_a_1186_);
lean_dec(v___x_1174_);
lean_dec(v___y_1170_);
lean_dec_ref(v___y_1169_);
lean_dec(v___y_1168_);
lean_dec_ref(v___y_1167_);
lean_dec(v___y_1164_);
lean_dec(v___y_1163_);
lean_dec(v_thm_1161_);
v_a_1368_ = lean_ctor_get(v___x_1365_, 0);
v_isSharedCheck_1375_ = !lean_is_exclusive(v___x_1365_);
if (v_isSharedCheck_1375_ == 0)
{
v___x_1370_ = v___x_1365_;
v_isShared_1371_ = v_isSharedCheck_1375_;
goto v_resetjp_1369_;
}
else
{
lean_inc(v_a_1368_);
lean_dec(v___x_1365_);
v___x_1370_ = lean_box(0);
v_isShared_1371_ = v_isSharedCheck_1375_;
goto v_resetjp_1369_;
}
v_resetjp_1369_:
{
lean_object* v___x_1373_; 
if (v_isShared_1371_ == 0)
{
v___x_1373_ = v___x_1370_;
goto v_reusejp_1372_;
}
else
{
lean_object* v_reuseFailAlloc_1374_; 
v_reuseFailAlloc_1374_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1374_, 0, v_a_1368_);
v___x_1373_ = v_reuseFailAlloc_1374_;
goto v_reusejp_1372_;
}
v_reusejp_1372_:
{
return v___x_1373_;
}
}
}
}
else
{
lean_object* v_a_1376_; lean_object* v___x_1378_; uint8_t v_isShared_1379_; uint8_t v_isSharedCheck_1383_; 
lean_del_object(v___x_1198_);
lean_dec(v_a_1192_);
lean_dec(v_a_1186_);
lean_dec(v___x_1174_);
lean_dec(v___y_1170_);
lean_dec_ref(v___y_1169_);
lean_dec(v___y_1168_);
lean_dec_ref(v___y_1167_);
lean_dec(v___y_1164_);
lean_dec(v___y_1163_);
lean_dec(v_thm_1161_);
v_a_1376_ = lean_ctor_get(v___x_1363_, 0);
v_isSharedCheck_1383_ = !lean_is_exclusive(v___x_1363_);
if (v_isSharedCheck_1383_ == 0)
{
v___x_1378_ = v___x_1363_;
v_isShared_1379_ = v_isSharedCheck_1383_;
goto v_resetjp_1377_;
}
else
{
lean_inc(v_a_1376_);
lean_dec(v___x_1363_);
v___x_1378_ = lean_box(0);
v_isShared_1379_ = v_isSharedCheck_1383_;
goto v_resetjp_1377_;
}
v_resetjp_1377_:
{
lean_object* v___x_1381_; 
if (v_isShared_1379_ == 0)
{
v___x_1381_ = v___x_1378_;
goto v_reusejp_1380_;
}
else
{
lean_object* v_reuseFailAlloc_1382_; 
v_reuseFailAlloc_1382_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1382_, 0, v_a_1376_);
v___x_1381_ = v_reuseFailAlloc_1382_;
goto v_reusejp_1380_;
}
v_reusejp_1380_:
{
return v___x_1381_;
}
}
}
}
else
{
lean_object* v_a_1384_; lean_object* v___x_1386_; uint8_t v_isShared_1387_; uint8_t v_isSharedCheck_1391_; 
lean_dec(v_snd_1360_);
lean_dec_ref_known(v___x_1335_, 0);
lean_del_object(v___x_1198_);
lean_dec(v_a_1192_);
lean_dec(v_a_1186_);
lean_dec(v___x_1174_);
lean_dec(v___y_1170_);
lean_dec_ref(v___y_1169_);
lean_dec(v___y_1168_);
lean_dec_ref(v___y_1167_);
lean_dec(v___y_1164_);
lean_dec(v___y_1163_);
lean_dec(v_thm_1161_);
v_a_1384_ = lean_ctor_get(v___x_1361_, 0);
v_isSharedCheck_1391_ = !lean_is_exclusive(v___x_1361_);
if (v_isSharedCheck_1391_ == 0)
{
v___x_1386_ = v___x_1361_;
v_isShared_1387_ = v_isSharedCheck_1391_;
goto v_resetjp_1385_;
}
else
{
lean_inc(v_a_1384_);
lean_dec(v___x_1361_);
v___x_1386_ = lean_box(0);
v_isShared_1387_ = v_isSharedCheck_1391_;
goto v_resetjp_1385_;
}
v_resetjp_1385_:
{
lean_object* v___x_1389_; 
if (v_isShared_1387_ == 0)
{
v___x_1389_ = v___x_1386_;
goto v_reusejp_1388_;
}
else
{
lean_object* v_reuseFailAlloc_1390_; 
v_reuseFailAlloc_1390_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1390_, 0, v_a_1384_);
v___x_1389_ = v_reuseFailAlloc_1390_;
goto v_reusejp_1388_;
}
v_reusejp_1388_:
{
return v___x_1389_;
}
}
}
}
else
{
lean_object* v_a_1392_; lean_object* v___x_1394_; uint8_t v_isShared_1395_; uint8_t v_isSharedCheck_1399_; 
lean_dec_ref_known(v___x_1335_, 0);
lean_del_object(v___x_1198_);
lean_dec(v_a_1192_);
lean_dec(v_a_1186_);
lean_dec(v___x_1174_);
lean_dec(v___y_1170_);
lean_dec_ref(v___y_1169_);
lean_dec(v___y_1168_);
lean_dec_ref(v___y_1167_);
lean_dec(v___y_1164_);
lean_dec(v___y_1163_);
lean_dec(v_thm_1161_);
v_a_1392_ = lean_ctor_get(v___x_1358_, 0);
v_isSharedCheck_1399_ = !lean_is_exclusive(v___x_1358_);
if (v_isSharedCheck_1399_ == 0)
{
v___x_1394_ = v___x_1358_;
v_isShared_1395_ = v_isSharedCheck_1399_;
goto v_resetjp_1393_;
}
else
{
lean_inc(v_a_1392_);
lean_dec(v___x_1358_);
v___x_1394_ = lean_box(0);
v_isShared_1395_ = v_isSharedCheck_1399_;
goto v_resetjp_1393_;
}
v_resetjp_1393_:
{
lean_object* v___x_1397_; 
if (v_isShared_1395_ == 0)
{
v___x_1397_ = v___x_1394_;
goto v_reusejp_1396_;
}
else
{
lean_object* v_reuseFailAlloc_1398_; 
v_reuseFailAlloc_1398_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1398_, 0, v_a_1392_);
v___x_1397_ = v_reuseFailAlloc_1398_;
goto v_reusejp_1396_;
}
v_reusejp_1396_:
{
return v___x_1397_;
}
}
}
}
else
{
lean_dec_ref_known(v_a_1338_, 2);
lean_dec_ref_known(v___x_1335_, 0);
lean_del_object(v___x_1198_);
lean_dec(v_a_1192_);
lean_dec(v_a_1186_);
lean_dec(v___x_1174_);
lean_dec(v___y_1164_);
lean_dec(v___y_1163_);
lean_dec(v_thm_1161_);
v___y_1340_ = v___y_1165_;
v___y_1341_ = v___y_1166_;
v___y_1342_ = v___y_1167_;
v___y_1343_ = v___y_1168_;
v___y_1344_ = v___y_1169_;
v___y_1345_ = v___y_1170_;
goto v___jp_1339_;
}
}
else
{
lean_dec(v_a_1338_);
lean_dec_ref_known(v___x_1335_, 0);
lean_del_object(v___x_1198_);
lean_dec(v_a_1192_);
lean_dec(v_a_1186_);
lean_dec(v___x_1174_);
lean_dec(v___y_1164_);
lean_dec(v___y_1163_);
lean_dec(v_thm_1161_);
v___y_1340_ = v___y_1165_;
v___y_1341_ = v___y_1166_;
v___y_1342_ = v___y_1167_;
v___y_1343_ = v___y_1168_;
v___y_1344_ = v___y_1169_;
v___y_1345_ = v___y_1170_;
goto v___jp_1339_;
}
v___jp_1339_:
{
lean_object* v___x_1346_; lean_object* v___x_1347_; lean_object* v_a_1348_; lean_object* v___x_1350_; uint8_t v_isShared_1351_; uint8_t v_isSharedCheck_1355_; 
v___x_1346_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_mkComp___closed__8, &lp_mathlib_Mathlib_Tactic_mkComp___closed__8_once, _init_lp_mathlib_Mathlib_Tactic_mkComp___closed__8);
v___x_1347_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_higherOrderGetParam_spec__7___redArg(v___x_1346_, v___y_1340_, v___y_1341_, v___y_1342_, v___y_1343_, v___y_1344_, v___y_1345_);
lean_dec(v___y_1345_);
lean_dec_ref(v___y_1344_);
lean_dec(v___y_1343_);
lean_dec_ref(v___y_1342_);
v_a_1348_ = lean_ctor_get(v___x_1347_, 0);
v_isSharedCheck_1355_ = !lean_is_exclusive(v___x_1347_);
if (v_isSharedCheck_1355_ == 0)
{
v___x_1350_ = v___x_1347_;
v_isShared_1351_ = v_isSharedCheck_1355_;
goto v_resetjp_1349_;
}
else
{
lean_inc(v_a_1348_);
lean_dec(v___x_1347_);
v___x_1350_ = lean_box(0);
v_isShared_1351_ = v_isSharedCheck_1355_;
goto v_resetjp_1349_;
}
v_resetjp_1349_:
{
lean_object* v___x_1353_; 
if (v_isShared_1351_ == 0)
{
v___x_1353_ = v___x_1350_;
goto v_reusejp_1352_;
}
else
{
lean_object* v_reuseFailAlloc_1354_; 
v_reuseFailAlloc_1354_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1354_, 0, v_a_1348_);
v___x_1353_ = v_reuseFailAlloc_1354_;
goto v_reusejp_1352_;
}
v_reusejp_1352_:
{
return v___x_1353_;
}
}
}
}
else
{
lean_object* v_a_1400_; lean_object* v___x_1402_; uint8_t v_isShared_1403_; uint8_t v_isSharedCheck_1407_; 
lean_dec_ref_known(v___x_1335_, 0);
lean_del_object(v___x_1198_);
lean_dec(v_a_1192_);
lean_dec(v_a_1186_);
lean_dec(v___x_1174_);
lean_dec(v___y_1170_);
lean_dec_ref(v___y_1169_);
lean_dec(v___y_1168_);
lean_dec_ref(v___y_1167_);
lean_dec(v___y_1164_);
lean_dec(v___y_1163_);
lean_dec(v_thm_1161_);
v_a_1400_ = lean_ctor_get(v___x_1337_, 0);
v_isSharedCheck_1407_ = !lean_is_exclusive(v___x_1337_);
if (v_isSharedCheck_1407_ == 0)
{
v___x_1402_ = v___x_1337_;
v_isShared_1403_ = v_isSharedCheck_1407_;
goto v_resetjp_1401_;
}
else
{
lean_inc(v_a_1400_);
lean_dec(v___x_1337_);
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
v___jp_1205_:
{
lean_object* v___x_1212_; lean_object* v___x_1213_; 
v___x_1212_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_higherOrderGetParam___lam__1___closed__3));
v___x_1213_ = l_Lean_Meta_getSimpExtension_x3f(v___x_1212_, v___y_1210_, v___y_1211_);
if (lean_obj_tag(v___x_1213_) == 0)
{
lean_object* v_a_1214_; lean_object* v___x_1216_; uint8_t v_isShared_1217_; uint8_t v_isSharedCheck_1251_; 
v_a_1214_ = lean_ctor_get(v___x_1213_, 0);
v_isSharedCheck_1251_ = !lean_is_exclusive(v___x_1213_);
if (v_isSharedCheck_1251_ == 0)
{
v___x_1216_ = v___x_1213_;
v_isShared_1217_ = v_isSharedCheck_1251_;
goto v_resetjp_1215_;
}
else
{
lean_inc(v_a_1214_);
lean_dec(v___x_1213_);
v___x_1216_ = lean_box(0);
v_isShared_1217_ = v_isSharedCheck_1251_;
goto v_resetjp_1215_;
}
v_resetjp_1215_:
{
if (lean_obj_tag(v_a_1214_) == 1)
{
lean_object* v_val_1218_; lean_object* v___x_1219_; lean_object* v_ext_1220_; lean_object* v_toEnvExtension_1221_; lean_object* v_env_1222_; lean_object* v_asyncMode_1223_; lean_object* v___x_1224_; lean_object* v_lemmaNames_1225_; uint8_t v___x_1226_; 
v_val_1218_ = lean_ctor_get(v_a_1214_, 0);
lean_inc(v_val_1218_);
lean_dec_ref_known(v_a_1214_, 1);
v___x_1219_ = lean_st_ref_get(v___y_1211_);
v_ext_1220_ = lean_ctor_get(v_val_1218_, 1);
v_toEnvExtension_1221_ = lean_ctor_get(v_ext_1220_, 0);
v_env_1222_ = lean_ctor_get(v___x_1219_, 0);
lean_inc_ref(v_env_1222_);
lean_dec(v___x_1219_);
v_asyncMode_1223_ = lean_ctor_get(v_toEnvExtension_1221_, 2);
v___x_1224_ = l_Lean_ScopedEnvExtension_getState___redArg(v___y_1207_, v_val_1218_, v_env_1222_, v_asyncMode_1223_);
v_lemmaNames_1225_ = lean_ctor_get(v___x_1224_, 2);
lean_inc_ref(v_lemmaNames_1225_);
lean_dec(v___x_1224_);
v___x_1226_ = lp_mathlib_Lean_PersistentHashMap_contains___at___00Mathlib_Tactic_higherOrderGetParam_spec__6___redArg(v_lemmaNames_1225_, v___y_1206_);
lean_dec_ref(v___y_1206_);
lean_dec_ref(v_lemmaNames_1225_);
if (v___x_1226_ == 0)
{
lean_object* v___x_1228_; 
lean_dec(v_val_1218_);
lean_dec(v___y_1211_);
lean_dec_ref(v___y_1210_);
lean_dec(v___y_1209_);
lean_dec_ref(v___y_1208_);
if (v_isShared_1217_ == 0)
{
lean_ctor_set(v___x_1216_, 0, v___y_1163_);
v___x_1228_ = v___x_1216_;
goto v_reusejp_1227_;
}
else
{
lean_object* v_reuseFailAlloc_1229_; 
v_reuseFailAlloc_1229_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1229_, 0, v___y_1163_);
v___x_1228_ = v_reuseFailAlloc_1229_;
goto v_reusejp_1227_;
}
v_reusejp_1227_:
{
return v___x_1228_;
}
}
else
{
uint8_t v___x_1230_; lean_object* v___x_1231_; lean_object* v___x_1232_; 
lean_del_object(v___x_1216_);
v___x_1230_ = 0;
v___x_1231_ = lean_unsigned_to_nat(1000u);
lean_inc(v___y_1163_);
v___x_1232_ = l_Lean_Meta_addSimpTheorem(v_val_1218_, v___y_1163_, v___x_1162_, v___x_1204_, v___x_1230_, v___x_1231_, v___y_1208_, v___y_1209_, v___y_1210_, v___y_1211_);
lean_dec(v___y_1211_);
lean_dec_ref(v___y_1210_);
lean_dec(v___y_1209_);
lean_dec_ref(v___y_1208_);
if (lean_obj_tag(v___x_1232_) == 0)
{
lean_object* v___x_1234_; uint8_t v_isShared_1235_; uint8_t v_isSharedCheck_1239_; 
v_isSharedCheck_1239_ = !lean_is_exclusive(v___x_1232_);
if (v_isSharedCheck_1239_ == 0)
{
lean_object* v_unused_1240_; 
v_unused_1240_ = lean_ctor_get(v___x_1232_, 0);
lean_dec(v_unused_1240_);
v___x_1234_ = v___x_1232_;
v_isShared_1235_ = v_isSharedCheck_1239_;
goto v_resetjp_1233_;
}
else
{
lean_dec(v___x_1232_);
v___x_1234_ = lean_box(0);
v_isShared_1235_ = v_isSharedCheck_1239_;
goto v_resetjp_1233_;
}
v_resetjp_1233_:
{
lean_object* v___x_1237_; 
if (v_isShared_1235_ == 0)
{
lean_ctor_set(v___x_1234_, 0, v___y_1163_);
v___x_1237_ = v___x_1234_;
goto v_reusejp_1236_;
}
else
{
lean_object* v_reuseFailAlloc_1238_; 
v_reuseFailAlloc_1238_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1238_, 0, v___y_1163_);
v___x_1237_ = v_reuseFailAlloc_1238_;
goto v_reusejp_1236_;
}
v_reusejp_1236_:
{
return v___x_1237_;
}
}
}
else
{
lean_object* v_a_1241_; lean_object* v___x_1243_; uint8_t v_isShared_1244_; uint8_t v_isSharedCheck_1248_; 
lean_dec(v___y_1163_);
v_a_1241_ = lean_ctor_get(v___x_1232_, 0);
v_isSharedCheck_1248_ = !lean_is_exclusive(v___x_1232_);
if (v_isSharedCheck_1248_ == 0)
{
v___x_1243_ = v___x_1232_;
v_isShared_1244_ = v_isSharedCheck_1248_;
goto v_resetjp_1242_;
}
else
{
lean_inc(v_a_1241_);
lean_dec(v___x_1232_);
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
}
else
{
lean_object* v___x_1249_; lean_object* v___x_1250_; 
lean_del_object(v___x_1216_);
lean_dec(v_a_1214_);
lean_dec_ref(v___y_1206_);
lean_dec(v___y_1163_);
v___x_1249_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_mkComp___closed__8, &lp_mathlib_Mathlib_Tactic_mkComp___closed__8_once, _init_lp_mathlib_Mathlib_Tactic_mkComp___closed__8);
v___x_1250_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_mkComp_spec__0___redArg(v___x_1249_, v___y_1208_, v___y_1209_, v___y_1210_, v___y_1211_);
lean_dec(v___y_1211_);
lean_dec_ref(v___y_1210_);
lean_dec(v___y_1209_);
lean_dec_ref(v___y_1208_);
return v___x_1250_;
}
}
}
else
{
lean_object* v_a_1252_; lean_object* v___x_1254_; uint8_t v_isShared_1255_; uint8_t v_isSharedCheck_1259_; 
lean_dec(v___y_1211_);
lean_dec_ref(v___y_1210_);
lean_dec(v___y_1209_);
lean_dec_ref(v___y_1208_);
lean_dec_ref(v___y_1206_);
lean_dec(v___y_1163_);
v_a_1252_ = lean_ctor_get(v___x_1213_, 0);
v_isSharedCheck_1259_ = !lean_is_exclusive(v___x_1213_);
if (v_isSharedCheck_1259_ == 0)
{
v___x_1254_ = v___x_1213_;
v_isShared_1255_ = v_isSharedCheck_1259_;
goto v_resetjp_1253_;
}
else
{
lean_inc(v_a_1252_);
lean_dec(v___x_1213_);
v___x_1254_ = lean_box(0);
v_isShared_1255_ = v_isSharedCheck_1259_;
goto v_resetjp_1253_;
}
v_resetjp_1253_:
{
lean_object* v___x_1257_; 
if (v_isShared_1255_ == 0)
{
v___x_1257_ = v___x_1254_;
goto v_reusejp_1256_;
}
else
{
lean_object* v_reuseFailAlloc_1258_; 
v_reuseFailAlloc_1258_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1258_, 0, v_a_1252_);
v___x_1257_ = v_reuseFailAlloc_1258_;
goto v_reusejp_1256_;
}
v_reusejp_1256_:
{
return v___x_1257_;
}
}
}
}
v___jp_1260_:
{
lean_object* v___x_1268_; lean_object* v___x_1270_; 
lean_inc_n(v___y_1163_, 2);
v___x_1268_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1268_, 0, v___y_1163_);
lean_ctor_set(v___x_1268_, 1, v___x_1174_);
lean_ctor_set(v___x_1268_, 2, v_a_1186_);
if (v_isShared_1199_ == 0)
{
lean_ctor_set_tag(v___x_1198_, 1);
lean_ctor_set(v___x_1198_, 1, v___x_1175_);
lean_ctor_set(v___x_1198_, 0, v___y_1163_);
v___x_1270_ = v___x_1198_;
goto v_reusejp_1269_;
}
else
{
lean_object* v_reuseFailAlloc_1334_; 
v_reuseFailAlloc_1334_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1334_, 0, v___y_1163_);
lean_ctor_set(v_reuseFailAlloc_1334_, 1, v___x_1175_);
v___x_1270_ = v_reuseFailAlloc_1334_;
goto v_reusejp_1269_;
}
v_reusejp_1269_:
{
lean_object* v___x_1271_; lean_object* v___x_1272_; lean_object* v___x_1273_; 
v___x_1271_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1271_, 0, v___x_1268_);
lean_ctor_set(v___x_1271_, 1, v_prf_1261_);
lean_ctor_set(v___x_1271_, 2, v___x_1270_);
v___x_1272_ = lean_alloc_ctor(2, 1, 0);
lean_ctor_set(v___x_1272_, 0, v___x_1271_);
v___x_1273_ = l_Lean_addDecl(v___x_1272_, v___x_1204_, v___y_1266_, v___y_1267_);
if (lean_obj_tag(v___x_1273_) == 0)
{
lean_object* v_ref_1274_; lean_object* v___x_1275_; 
lean_dec_ref_known(v___x_1273_, 1);
v_ref_1274_ = lean_ctor_get(v___y_1266_, 5);
lean_inc(v___y_1163_);
v___x_1275_ = lp_mathlib_Lean_Elab_addDeclarationRangesFromSyntax___at___00Mathlib_Tactic_higherOrderGetParam_spec__4(v___y_1163_, v_ref_1274_, v___y_1164_, v___y_1262_, v___y_1263_, v___y_1264_, v___y_1265_, v___y_1266_, v___y_1267_);
if (lean_obj_tag(v___x_1275_) == 0)
{
lean_object* v___x_1276_; 
lean_dec_ref_known(v___x_1275_, 1);
lean_inc(v___y_1163_);
v___x_1276_ = lp_mathlib_Lean_mkConstWithLevelParams___at___00Mathlib_Tactic_higherOrderGetParam_spec__5(v___y_1163_, v___y_1262_, v___y_1263_, v___y_1264_, v___y_1265_, v___y_1266_, v___y_1267_);
if (lean_obj_tag(v___x_1276_) == 0)
{
lean_object* v_a_1277_; lean_object* v___x_1278_; lean_object* v___x_1279_; 
v_a_1277_ = lean_ctor_get(v___x_1276_, 0);
lean_inc(v_a_1277_);
lean_dec_ref_known(v___x_1276_, 1);
v___x_1278_ = lean_box(0);
v___x_1279_ = l_Lean_Elab_Term_addTermInfo_x27(v___y_1164_, v_a_1277_, v___x_1278_, v___x_1278_, v___x_1190_, v___x_1162_, v___x_1204_, v___y_1262_, v___y_1263_, v___y_1264_, v___y_1265_, v___y_1266_, v___y_1267_);
if (lean_obj_tag(v___x_1279_) == 0)
{
lean_object* v___x_1280_; lean_object* v_env_1281_; lean_object* v___x_1282_; lean_object* v_ext_1283_; lean_object* v_toEnvExtension_1284_; lean_object* v_asyncMode_1285_; lean_object* v___x_1286_; lean_object* v___x_1287_; lean_object* v_lemmaNames_1288_; lean_object* v___x_1289_; uint8_t v___x_1290_; 
lean_dec_ref_known(v___x_1279_, 1);
v___x_1280_ = lean_st_ref_get(v___y_1267_);
v_env_1281_ = lean_ctor_get(v___x_1280_, 0);
lean_inc_ref(v_env_1281_);
lean_dec(v___x_1280_);
v___x_1282_ = l_Lean_Meta_simpExtension;
v_ext_1283_ = lean_ctor_get(v___x_1282_, 1);
v_toEnvExtension_1284_ = lean_ctor_get(v_ext_1283_, 0);
v_asyncMode_1285_ = lean_ctor_get(v_toEnvExtension_1284_, 2);
v___x_1286_ = l_Lean_Meta_instInhabitedSimpTheorems_default;
v___x_1287_ = l_Lean_ScopedEnvExtension_getState___redArg(v___x_1286_, v___x_1282_, v_env_1281_, v_asyncMode_1285_);
v_lemmaNames_1288_ = lean_ctor_get(v___x_1287_, 2);
lean_inc_ref(v_lemmaNames_1288_);
lean_dec(v___x_1287_);
v___x_1289_ = lean_alloc_ctor(0, 1, 2);
lean_ctor_set(v___x_1289_, 0, v_thm_1161_);
lean_ctor_set_uint8(v___x_1289_, sizeof(void*)*1, v___x_1162_);
lean_ctor_set_uint8(v___x_1289_, sizeof(void*)*1 + 1, v___x_1204_);
v___x_1290_ = lp_mathlib_Lean_PersistentHashMap_contains___at___00Mathlib_Tactic_higherOrderGetParam_spec__6___redArg(v_lemmaNames_1288_, v___x_1289_);
lean_dec_ref(v_lemmaNames_1288_);
if (v___x_1290_ == 0)
{
v___y_1206_ = v___x_1289_;
v___y_1207_ = v___x_1286_;
v___y_1208_ = v___y_1264_;
v___y_1209_ = v___y_1265_;
v___y_1210_ = v___y_1266_;
v___y_1211_ = v___y_1267_;
goto v___jp_1205_;
}
else
{
uint8_t v___x_1291_; lean_object* v___x_1292_; lean_object* v___x_1293_; 
v___x_1291_ = 0;
v___x_1292_ = lean_unsigned_to_nat(1000u);
lean_inc(v___y_1163_);
v___x_1293_ = l_Lean_Meta_addSimpTheorem(v___x_1282_, v___y_1163_, v___x_1162_, v___x_1204_, v___x_1291_, v___x_1292_, v___y_1264_, v___y_1265_, v___y_1266_, v___y_1267_);
if (lean_obj_tag(v___x_1293_) == 0)
{
lean_dec_ref_known(v___x_1293_, 1);
v___y_1206_ = v___x_1289_;
v___y_1207_ = v___x_1286_;
v___y_1208_ = v___y_1264_;
v___y_1209_ = v___y_1265_;
v___y_1210_ = v___y_1266_;
v___y_1211_ = v___y_1267_;
goto v___jp_1205_;
}
else
{
lean_object* v_a_1294_; lean_object* v___x_1296_; uint8_t v_isShared_1297_; uint8_t v_isSharedCheck_1301_; 
lean_dec_ref_known(v___x_1289_, 1);
lean_dec(v___y_1267_);
lean_dec_ref(v___y_1266_);
lean_dec(v___y_1265_);
lean_dec_ref(v___y_1264_);
lean_dec(v___y_1163_);
v_a_1294_ = lean_ctor_get(v___x_1293_, 0);
v_isSharedCheck_1301_ = !lean_is_exclusive(v___x_1293_);
if (v_isSharedCheck_1301_ == 0)
{
v___x_1296_ = v___x_1293_;
v_isShared_1297_ = v_isSharedCheck_1301_;
goto v_resetjp_1295_;
}
else
{
lean_inc(v_a_1294_);
lean_dec(v___x_1293_);
v___x_1296_ = lean_box(0);
v_isShared_1297_ = v_isSharedCheck_1301_;
goto v_resetjp_1295_;
}
v_resetjp_1295_:
{
lean_object* v___x_1299_; 
if (v_isShared_1297_ == 0)
{
v___x_1299_ = v___x_1296_;
goto v_reusejp_1298_;
}
else
{
lean_object* v_reuseFailAlloc_1300_; 
v_reuseFailAlloc_1300_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1300_, 0, v_a_1294_);
v___x_1299_ = v_reuseFailAlloc_1300_;
goto v_reusejp_1298_;
}
v_reusejp_1298_:
{
return v___x_1299_;
}
}
}
}
}
else
{
lean_object* v_a_1302_; lean_object* v___x_1304_; uint8_t v_isShared_1305_; uint8_t v_isSharedCheck_1309_; 
lean_dec(v___y_1267_);
lean_dec_ref(v___y_1266_);
lean_dec(v___y_1265_);
lean_dec_ref(v___y_1264_);
lean_dec(v___y_1163_);
lean_dec(v_thm_1161_);
v_a_1302_ = lean_ctor_get(v___x_1279_, 0);
v_isSharedCheck_1309_ = !lean_is_exclusive(v___x_1279_);
if (v_isSharedCheck_1309_ == 0)
{
v___x_1304_ = v___x_1279_;
v_isShared_1305_ = v_isSharedCheck_1309_;
goto v_resetjp_1303_;
}
else
{
lean_inc(v_a_1302_);
lean_dec(v___x_1279_);
v___x_1304_ = lean_box(0);
v_isShared_1305_ = v_isSharedCheck_1309_;
goto v_resetjp_1303_;
}
v_resetjp_1303_:
{
lean_object* v___x_1307_; 
if (v_isShared_1305_ == 0)
{
v___x_1307_ = v___x_1304_;
goto v_reusejp_1306_;
}
else
{
lean_object* v_reuseFailAlloc_1308_; 
v_reuseFailAlloc_1308_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1308_, 0, v_a_1302_);
v___x_1307_ = v_reuseFailAlloc_1308_;
goto v_reusejp_1306_;
}
v_reusejp_1306_:
{
return v___x_1307_;
}
}
}
}
else
{
lean_object* v_a_1310_; lean_object* v___x_1312_; uint8_t v_isShared_1313_; uint8_t v_isSharedCheck_1317_; 
lean_dec(v___y_1267_);
lean_dec_ref(v___y_1266_);
lean_dec(v___y_1265_);
lean_dec_ref(v___y_1264_);
lean_dec(v___y_1164_);
lean_dec(v___y_1163_);
lean_dec(v_thm_1161_);
v_a_1310_ = lean_ctor_get(v___x_1276_, 0);
v_isSharedCheck_1317_ = !lean_is_exclusive(v___x_1276_);
if (v_isSharedCheck_1317_ == 0)
{
v___x_1312_ = v___x_1276_;
v_isShared_1313_ = v_isSharedCheck_1317_;
goto v_resetjp_1311_;
}
else
{
lean_inc(v_a_1310_);
lean_dec(v___x_1276_);
v___x_1312_ = lean_box(0);
v_isShared_1313_ = v_isSharedCheck_1317_;
goto v_resetjp_1311_;
}
v_resetjp_1311_:
{
lean_object* v___x_1315_; 
if (v_isShared_1313_ == 0)
{
v___x_1315_ = v___x_1312_;
goto v_reusejp_1314_;
}
else
{
lean_object* v_reuseFailAlloc_1316_; 
v_reuseFailAlloc_1316_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1316_, 0, v_a_1310_);
v___x_1315_ = v_reuseFailAlloc_1316_;
goto v_reusejp_1314_;
}
v_reusejp_1314_:
{
return v___x_1315_;
}
}
}
}
else
{
lean_object* v_a_1318_; lean_object* v___x_1320_; uint8_t v_isShared_1321_; uint8_t v_isSharedCheck_1325_; 
lean_dec(v___y_1267_);
lean_dec_ref(v___y_1266_);
lean_dec(v___y_1265_);
lean_dec_ref(v___y_1264_);
lean_dec(v___y_1164_);
lean_dec(v___y_1163_);
lean_dec(v_thm_1161_);
v_a_1318_ = lean_ctor_get(v___x_1275_, 0);
v_isSharedCheck_1325_ = !lean_is_exclusive(v___x_1275_);
if (v_isSharedCheck_1325_ == 0)
{
v___x_1320_ = v___x_1275_;
v_isShared_1321_ = v_isSharedCheck_1325_;
goto v_resetjp_1319_;
}
else
{
lean_inc(v_a_1318_);
lean_dec(v___x_1275_);
v___x_1320_ = lean_box(0);
v_isShared_1321_ = v_isSharedCheck_1325_;
goto v_resetjp_1319_;
}
v_resetjp_1319_:
{
lean_object* v___x_1323_; 
if (v_isShared_1321_ == 0)
{
v___x_1323_ = v___x_1320_;
goto v_reusejp_1322_;
}
else
{
lean_object* v_reuseFailAlloc_1324_; 
v_reuseFailAlloc_1324_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1324_, 0, v_a_1318_);
v___x_1323_ = v_reuseFailAlloc_1324_;
goto v_reusejp_1322_;
}
v_reusejp_1322_:
{
return v___x_1323_;
}
}
}
}
else
{
lean_object* v_a_1326_; lean_object* v___x_1328_; uint8_t v_isShared_1329_; uint8_t v_isSharedCheck_1333_; 
lean_dec(v___y_1267_);
lean_dec_ref(v___y_1266_);
lean_dec(v___y_1265_);
lean_dec_ref(v___y_1264_);
lean_dec(v___y_1164_);
lean_dec(v___y_1163_);
lean_dec(v_thm_1161_);
v_a_1326_ = lean_ctor_get(v___x_1273_, 0);
v_isSharedCheck_1333_ = !lean_is_exclusive(v___x_1273_);
if (v_isSharedCheck_1333_ == 0)
{
v___x_1328_ = v___x_1273_;
v_isShared_1329_ = v_isSharedCheck_1333_;
goto v_resetjp_1327_;
}
else
{
lean_inc(v_a_1326_);
lean_dec(v___x_1273_);
v___x_1328_ = lean_box(0);
v_isShared_1329_ = v_isSharedCheck_1333_;
goto v_resetjp_1327_;
}
v_resetjp_1327_:
{
lean_object* v___x_1331_; 
if (v_isShared_1329_ == 0)
{
v___x_1331_ = v___x_1328_;
goto v_reusejp_1330_;
}
else
{
lean_object* v_reuseFailAlloc_1332_; 
v_reuseFailAlloc_1332_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1332_, 0, v_a_1326_);
v___x_1331_ = v_reuseFailAlloc_1332_;
goto v_reusejp_1330_;
}
v_reusejp_1330_:
{
return v___x_1331_;
}
}
}
}
}
}
else
{
lean_object* v_a_1408_; lean_object* v___x_1410_; uint8_t v_isShared_1411_; uint8_t v_isSharedCheck_1415_; 
lean_del_object(v___x_1198_);
lean_dec(v_snd_1196_);
lean_dec(v_a_1192_);
lean_dec(v_a_1186_);
lean_dec(v___x_1174_);
lean_dec(v___y_1170_);
lean_dec_ref(v___y_1169_);
lean_dec(v___y_1168_);
lean_dec_ref(v___y_1167_);
lean_dec(v___y_1164_);
lean_dec(v___y_1163_);
lean_dec(v_thm_1161_);
v_a_1408_ = lean_ctor_get(v___x_1201_, 0);
v_isSharedCheck_1415_ = !lean_is_exclusive(v___x_1201_);
if (v_isSharedCheck_1415_ == 0)
{
v___x_1410_ = v___x_1201_;
v_isShared_1411_ = v_isSharedCheck_1415_;
goto v_resetjp_1409_;
}
else
{
lean_inc(v_a_1408_);
lean_dec(v___x_1201_);
v___x_1410_ = lean_box(0);
v_isShared_1411_ = v_isSharedCheck_1415_;
goto v_resetjp_1409_;
}
v_resetjp_1409_:
{
lean_object* v___x_1413_; 
if (v_isShared_1411_ == 0)
{
v___x_1413_ = v___x_1410_;
goto v_reusejp_1412_;
}
else
{
lean_object* v_reuseFailAlloc_1414_; 
v_reuseFailAlloc_1414_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1414_, 0, v_a_1408_);
v___x_1413_ = v_reuseFailAlloc_1414_;
goto v_reusejp_1412_;
}
v_reusejp_1412_:
{
return v___x_1413_;
}
}
}
}
}
else
{
lean_object* v_a_1418_; lean_object* v___x_1420_; uint8_t v_isShared_1421_; uint8_t v_isSharedCheck_1425_; 
lean_dec(v_a_1192_);
lean_dec(v_a_1186_);
lean_dec(v___x_1174_);
lean_dec(v___y_1170_);
lean_dec_ref(v___y_1169_);
lean_dec(v___y_1168_);
lean_dec_ref(v___y_1167_);
lean_dec(v___y_1164_);
lean_dec(v___y_1163_);
lean_dec(v_thm_1161_);
v_a_1418_ = lean_ctor_get(v___x_1194_, 0);
v_isSharedCheck_1425_ = !lean_is_exclusive(v___x_1194_);
if (v_isSharedCheck_1425_ == 0)
{
v___x_1420_ = v___x_1194_;
v_isShared_1421_ = v_isSharedCheck_1425_;
goto v_resetjp_1419_;
}
else
{
lean_inc(v_a_1418_);
lean_dec(v___x_1194_);
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
lean_dec(v_a_1186_);
lean_dec(v___x_1174_);
lean_dec(v___y_1170_);
lean_dec_ref(v___y_1169_);
lean_dec(v___y_1168_);
lean_dec_ref(v___y_1167_);
lean_dec(v___y_1164_);
lean_dec(v___y_1163_);
lean_dec(v_thm_1161_);
v_a_1426_ = lean_ctor_get(v___x_1191_, 0);
v_isSharedCheck_1433_ = !lean_is_exclusive(v___x_1191_);
if (v_isSharedCheck_1433_ == 0)
{
v___x_1428_ = v___x_1191_;
v_isShared_1429_ = v_isSharedCheck_1433_;
goto v_resetjp_1427_;
}
else
{
lean_inc(v_a_1426_);
lean_dec(v___x_1191_);
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
}
else
{
lean_object* v_a_1435_; lean_object* v___x_1437_; uint8_t v_isShared_1438_; uint8_t v_isSharedCheck_1442_; 
lean_del_object(v___x_1183_);
lean_dec(v___x_1174_);
lean_dec(v___y_1170_);
lean_dec_ref(v___y_1169_);
lean_dec(v___y_1168_);
lean_dec_ref(v___y_1167_);
lean_dec(v___y_1164_);
lean_dec(v___y_1163_);
lean_dec(v_thm_1161_);
v_a_1435_ = lean_ctor_get(v___x_1185_, 0);
v_isSharedCheck_1442_ = !lean_is_exclusive(v___x_1185_);
if (v_isSharedCheck_1442_ == 0)
{
v___x_1437_ = v___x_1185_;
v_isShared_1438_ = v_isSharedCheck_1442_;
goto v_resetjp_1436_;
}
else
{
lean_inc(v_a_1435_);
lean_dec(v___x_1185_);
v___x_1437_ = lean_box(0);
v_isShared_1438_ = v_isSharedCheck_1442_;
goto v_resetjp_1436_;
}
v_resetjp_1436_:
{
lean_object* v___x_1440_; 
if (v_isShared_1438_ == 0)
{
v___x_1440_ = v___x_1437_;
goto v_reusejp_1439_;
}
else
{
lean_object* v_reuseFailAlloc_1441_; 
v_reuseFailAlloc_1441_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1441_, 0, v_a_1435_);
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
else
{
lean_object* v_a_1444_; lean_object* v___x_1446_; uint8_t v_isShared_1447_; uint8_t v_isSharedCheck_1451_; 
lean_dec(v___x_1174_);
lean_dec(v___y_1170_);
lean_dec_ref(v___y_1169_);
lean_dec(v___y_1168_);
lean_dec_ref(v___y_1167_);
lean_dec(v___y_1164_);
lean_dec(v___y_1163_);
lean_dec(v_thm_1161_);
v_a_1444_ = lean_ctor_get(v___x_1178_, 0);
v_isSharedCheck_1451_ = !lean_is_exclusive(v___x_1178_);
if (v_isSharedCheck_1451_ == 0)
{
v___x_1446_ = v___x_1178_;
v_isShared_1447_ = v_isSharedCheck_1451_;
goto v_resetjp_1445_;
}
else
{
lean_inc(v_a_1444_);
lean_dec(v___x_1178_);
v___x_1446_ = lean_box(0);
v_isShared_1447_ = v_isSharedCheck_1451_;
goto v_resetjp_1445_;
}
v_resetjp_1445_:
{
lean_object* v___x_1449_; 
if (v_isShared_1447_ == 0)
{
v___x_1449_ = v___x_1446_;
goto v_reusejp_1448_;
}
else
{
lean_object* v_reuseFailAlloc_1450_; 
v_reuseFailAlloc_1450_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1450_, 0, v_a_1444_);
v___x_1449_ = v_reuseFailAlloc_1450_;
goto v_reusejp_1448_;
}
v_reusejp_1448_:
{
return v___x_1449_;
}
}
}
}
else
{
lean_object* v_a_1452_; lean_object* v___x_1454_; uint8_t v_isShared_1455_; uint8_t v_isSharedCheck_1459_; 
lean_dec(v___y_1170_);
lean_dec_ref(v___y_1169_);
lean_dec(v___y_1168_);
lean_dec_ref(v___y_1167_);
lean_dec(v___y_1164_);
lean_dec(v___y_1163_);
lean_dec(v_thm_1161_);
v_a_1452_ = lean_ctor_get(v___x_1172_, 0);
v_isSharedCheck_1459_ = !lean_is_exclusive(v___x_1172_);
if (v_isSharedCheck_1459_ == 0)
{
v___x_1454_ = v___x_1172_;
v_isShared_1455_ = v_isSharedCheck_1459_;
goto v_resetjp_1453_;
}
else
{
lean_inc(v_a_1452_);
lean_dec(v___x_1172_);
v___x_1454_ = lean_box(0);
v_isShared_1455_ = v_isSharedCheck_1459_;
goto v_resetjp_1453_;
}
v_resetjp_1453_:
{
lean_object* v___x_1457_; 
if (v_isShared_1455_ == 0)
{
v___x_1457_ = v___x_1454_;
goto v_reusejp_1456_;
}
else
{
lean_object* v_reuseFailAlloc_1458_; 
v_reuseFailAlloc_1458_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1458_, 0, v_a_1452_);
v___x_1457_ = v_reuseFailAlloc_1458_;
goto v_reusejp_1456_;
}
v_reusejp_1456_:
{
return v___x_1457_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_higherOrderGetParam___lam__1___boxed(lean_object* v_thm_1460_, lean_object* v___x_1461_, lean_object* v___y_1462_, lean_object* v___y_1463_, lean_object* v___y_1464_, lean_object* v___y_1465_, lean_object* v___y_1466_, lean_object* v___y_1467_, lean_object* v___y_1468_, lean_object* v___y_1469_, lean_object* v___y_1470_){
_start:
{
uint8_t v___x_19240__boxed_1471_; lean_object* v_res_1472_; 
v___x_19240__boxed_1471_ = lean_unbox(v___x_1461_);
v_res_1472_ = lp_mathlib_Mathlib_Tactic_higherOrderGetParam___lam__1(v_thm_1460_, v___x_19240__boxed_1471_, v___y_1462_, v___y_1463_, v___y_1464_, v___y_1465_, v___y_1466_, v___y_1467_, v___y_1468_, v___y_1469_);
lean_dec(v___y_1465_);
lean_dec_ref(v___y_1464_);
return v_res_1472_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_higherOrderGetParam___closed__2(void){
_start:
{
lean_object* v___x_1476_; 
v___x_1476_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_1476_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_higherOrderGetParam___closed__3(void){
_start:
{
lean_object* v___x_1477_; lean_object* v___x_1478_; 
v___x_1477_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_higherOrderGetParam___closed__2, &lp_mathlib_Mathlib_Tactic_higherOrderGetParam___closed__2_once, _init_lp_mathlib_Mathlib_Tactic_higherOrderGetParam___closed__2);
v___x_1478_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1478_, 0, v___x_1477_);
return v___x_1478_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_higherOrderGetParam___closed__4(void){
_start:
{
lean_object* v___x_1479_; lean_object* v___x_1480_; lean_object* v___x_1481_; lean_object* v___x_1482_; 
v___x_1479_ = lean_box(1);
v___x_1480_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3_spec__12_spec__17_spec__20___redArg___closed__4, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3_spec__12_spec__17_spec__20___redArg___closed__4_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3_spec__12_spec__17_spec__20___redArg___closed__4);
v___x_1481_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_higherOrderGetParam___closed__3, &lp_mathlib_Mathlib_Tactic_higherOrderGetParam___closed__3_once, _init_lp_mathlib_Mathlib_Tactic_higherOrderGetParam___closed__3);
v___x_1482_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1482_, 0, v___x_1481_);
lean_ctor_set(v___x_1482_, 1, v___x_1480_);
lean_ctor_set(v___x_1482_, 2, v___x_1479_);
return v___x_1482_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_higherOrderGetParam___closed__5(void){
_start:
{
lean_object* v___x_1483_; lean_object* v___x_1484_; lean_object* v___x_1485_; 
v___x_1483_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_higherOrderGetParam___closed__3, &lp_mathlib_Mathlib_Tactic_higherOrderGetParam___closed__3_once, _init_lp_mathlib_Mathlib_Tactic_higherOrderGetParam___closed__3);
v___x_1484_ = lean_unsigned_to_nat(0u);
v___x_1485_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v___x_1485_, 0, v___x_1484_);
lean_ctor_set(v___x_1485_, 1, v___x_1484_);
lean_ctor_set(v___x_1485_, 2, v___x_1484_);
lean_ctor_set(v___x_1485_, 3, v___x_1484_);
lean_ctor_set(v___x_1485_, 4, v___x_1483_);
lean_ctor_set(v___x_1485_, 5, v___x_1483_);
lean_ctor_set(v___x_1485_, 6, v___x_1483_);
lean_ctor_set(v___x_1485_, 7, v___x_1483_);
lean_ctor_set(v___x_1485_, 8, v___x_1483_);
lean_ctor_set(v___x_1485_, 9, v___x_1483_);
return v___x_1485_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_higherOrderGetParam___closed__6(void){
_start:
{
lean_object* v___x_1486_; lean_object* v___x_1487_; 
v___x_1486_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_higherOrderGetParam___closed__3, &lp_mathlib_Mathlib_Tactic_higherOrderGetParam___closed__3_once, _init_lp_mathlib_Mathlib_Tactic_higherOrderGetParam___closed__3);
v___x_1487_ = lean_alloc_ctor(0, 6, 0);
lean_ctor_set(v___x_1487_, 0, v___x_1486_);
lean_ctor_set(v___x_1487_, 1, v___x_1486_);
lean_ctor_set(v___x_1487_, 2, v___x_1486_);
lean_ctor_set(v___x_1487_, 3, v___x_1486_);
lean_ctor_set(v___x_1487_, 4, v___x_1486_);
lean_ctor_set(v___x_1487_, 5, v___x_1486_);
return v___x_1487_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_higherOrderGetParam___closed__7(void){
_start:
{
lean_object* v___x_1488_; lean_object* v___x_1489_; 
v___x_1488_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_higherOrderGetParam___closed__3, &lp_mathlib_Mathlib_Tactic_higherOrderGetParam___closed__3_once, _init_lp_mathlib_Mathlib_Tactic_higherOrderGetParam___closed__3);
v___x_1489_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_1489_, 0, v___x_1488_);
lean_ctor_set(v___x_1489_, 1, v___x_1488_);
lean_ctor_set(v___x_1489_, 2, v___x_1488_);
lean_ctor_set(v___x_1489_, 3, v___x_1488_);
lean_ctor_set(v___x_1489_, 4, v___x_1488_);
return v___x_1489_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_higherOrderGetParam___closed__8(void){
_start:
{
lean_object* v___x_1490_; lean_object* v___x_1491_; lean_object* v___x_1492_; lean_object* v___x_1493_; lean_object* v___x_1494_; lean_object* v___x_1495_; 
v___x_1490_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_higherOrderGetParam___closed__7, &lp_mathlib_Mathlib_Tactic_higherOrderGetParam___closed__7_once, _init_lp_mathlib_Mathlib_Tactic_higherOrderGetParam___closed__7);
v___x_1491_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3_spec__12_spec__17_spec__20___redArg___closed__4, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3_spec__12_spec__17_spec__20___redArg___closed__4_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3_spec__12_spec__17_spec__20___redArg___closed__4);
v___x_1492_ = lean_box(1);
v___x_1493_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_higherOrderGetParam___closed__6, &lp_mathlib_Mathlib_Tactic_higherOrderGetParam___closed__6_once, _init_lp_mathlib_Mathlib_Tactic_higherOrderGetParam___closed__6);
v___x_1494_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_higherOrderGetParam___closed__5, &lp_mathlib_Mathlib_Tactic_higherOrderGetParam___closed__5_once, _init_lp_mathlib_Mathlib_Tactic_higherOrderGetParam___closed__5);
v___x_1495_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_1495_, 0, v___x_1494_);
lean_ctor_set(v___x_1495_, 1, v___x_1493_);
lean_ctor_set(v___x_1495_, 2, v___x_1492_);
lean_ctor_set(v___x_1495_, 3, v___x_1491_);
lean_ctor_set(v___x_1495_, 4, v___x_1490_);
return v___x_1495_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_higherOrderGetParam(lean_object* v_thm_1500_, lean_object* v_stx_1501_, lean_object* v_a_1502_, lean_object* v_a_1503_){
_start:
{
lean_object* v___x_1505_; uint8_t v___x_1506_; 
v___x_1505_ = ((lean_object*)(lp_mathlib_Lean_Parser_Attr_higherOrder___closed__4));
lean_inc(v_stx_1501_);
v___x_1506_ = l_Lean_Syntax_isOfKind(v_stx_1501_, v___x_1505_);
if (v___x_1506_ == 0)
{
lean_object* v___x_1507_; 
lean_dec(v_stx_1501_);
lean_dec(v_thm_1500_);
v___x_1507_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_higherOrderGetParam_spec__0___redArg();
return v___x_1507_;
}
else
{
lean_object* v___f_1508_; lean_object* v___x_1509_; lean_object* v___y_1511_; lean_object* v___y_1512_; lean_object* v___y_1513_; lean_object* v___y_1514_; lean_object* v___x_1553_; lean_object* v___x_1554_; uint8_t v___x_1555_; 
v___f_1508_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_higherOrderGetParam___closed__0));
v___x_1509_ = lean_unsigned_to_nat(0u);
v___x_1553_ = lean_unsigned_to_nat(1u);
v___x_1554_ = l_Lean_Syntax_getArg(v_stx_1501_, v___x_1553_);
v___x_1555_ = l_Lean_Syntax_isNone(v___x_1554_);
if (v___x_1555_ == 0)
{
uint8_t v___x_1556_; 
lean_dec(v_stx_1501_);
lean_inc(v___x_1554_);
v___x_1556_ = l_Lean_Syntax_matchesNull(v___x_1554_, v___x_1553_);
if (v___x_1556_ == 0)
{
lean_object* v___x_1557_; 
lean_dec(v___x_1554_);
lean_dec(v_thm_1500_);
v___x_1557_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_higherOrderGetParam_spec__0___redArg();
return v___x_1557_;
}
else
{
lean_object* v_name_1558_; lean_object* v___x_1559_; lean_object* v___x_1560_; lean_object* v___x_1561_; 
v_name_1558_ = l_Lean_Syntax_getArg(v___x_1554_, v___x_1509_);
lean_dec(v___x_1554_);
v___x_1559_ = l_Lean_TSyntax_getId(v_name_1558_);
v___x_1560_ = l_Lean_Name_getPrefix(v_thm_1500_);
v___x_1561_ = l_Lean_Name_updatePrefix(v___x_1559_, v___x_1560_);
v___y_1511_ = v_a_1503_;
v___y_1512_ = v_a_1502_;
v___y_1513_ = v_name_1558_;
v___y_1514_ = v___x_1561_;
goto v___jp_1510_;
}
}
else
{
lean_object* v___x_1562_; lean_object* v___x_1563_; lean_object* v___x_1564_; 
lean_dec(v___x_1554_);
v___x_1562_ = l_Lean_Syntax_getArg(v_stx_1501_, v___x_1509_);
lean_dec(v_stx_1501_);
v___x_1563_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_higherOrderGetParam___closed__10));
lean_inc(v_thm_1500_);
v___x_1564_ = lean_name_append_after(v_thm_1500_, v___x_1563_);
v___y_1511_ = v_a_1503_;
v___y_1512_ = v_a_1502_;
v___y_1513_ = v___x_1562_;
v___y_1514_ = v___x_1564_;
goto v___jp_1510_;
}
v___jp_1510_:
{
lean_object* v___x_1515_; lean_object* v___x_1516_; lean_object* v___x_1517_; uint8_t v___x_1518_; lean_object* v___x_1519_; lean_object* v___x_1520_; uint8_t v___x_1521_; uint8_t v___x_1522_; uint8_t v___x_1523_; lean_object* v___x_1524_; uint64_t v___x_1525_; lean_object* v___x_1526_; lean_object* v___x_1527_; lean_object* v___x_1528_; lean_object* v___x_1529_; lean_object* v___x_1530_; lean_object* v___x_1531_; lean_object* v___f_1532_; lean_object* v___x_1533_; lean_object* v___x_1534_; 
v___x_1515_ = lean_box(0);
v___x_1516_ = lean_box(0);
v___x_1517_ = lean_box(1);
v___x_1518_ = 0;
v___x_1519_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_higherOrderGetParam___closed__1));
v___x_1520_ = lean_alloc_ctor(0, 8, 11);
lean_ctor_set(v___x_1520_, 0, v___x_1515_);
lean_ctor_set(v___x_1520_, 1, v___x_1516_);
lean_ctor_set(v___x_1520_, 2, v___x_1515_);
lean_ctor_set(v___x_1520_, 3, v___f_1508_);
lean_ctor_set(v___x_1520_, 4, v___x_1517_);
lean_ctor_set(v___x_1520_, 5, v___x_1517_);
lean_ctor_set(v___x_1520_, 6, v___x_1515_);
lean_ctor_set(v___x_1520_, 7, v___x_1519_);
lean_ctor_set_uint8(v___x_1520_, sizeof(void*)*8, v___x_1506_);
lean_ctor_set_uint8(v___x_1520_, sizeof(void*)*8 + 1, v___x_1506_);
lean_ctor_set_uint8(v___x_1520_, sizeof(void*)*8 + 2, v___x_1506_);
lean_ctor_set_uint8(v___x_1520_, sizeof(void*)*8 + 3, v___x_1506_);
lean_ctor_set_uint8(v___x_1520_, sizeof(void*)*8 + 4, v___x_1518_);
lean_ctor_set_uint8(v___x_1520_, sizeof(void*)*8 + 5, v___x_1518_);
lean_ctor_set_uint8(v___x_1520_, sizeof(void*)*8 + 6, v___x_1518_);
lean_ctor_set_uint8(v___x_1520_, sizeof(void*)*8 + 7, v___x_1518_);
lean_ctor_set_uint8(v___x_1520_, sizeof(void*)*8 + 8, v___x_1506_);
lean_ctor_set_uint8(v___x_1520_, sizeof(void*)*8 + 9, v___x_1518_);
lean_ctor_set_uint8(v___x_1520_, sizeof(void*)*8 + 10, v___x_1506_);
v___x_1521_ = 1;
v___x_1522_ = 0;
v___x_1523_ = 2;
v___x_1524_ = lean_alloc_ctor(0, 0, 20);
lean_ctor_set_uint8(v___x_1524_, 0, v___x_1518_);
lean_ctor_set_uint8(v___x_1524_, 1, v___x_1518_);
lean_ctor_set_uint8(v___x_1524_, 2, v___x_1518_);
lean_ctor_set_uint8(v___x_1524_, 3, v___x_1518_);
lean_ctor_set_uint8(v___x_1524_, 4, v___x_1518_);
lean_ctor_set_uint8(v___x_1524_, 5, v___x_1506_);
lean_ctor_set_uint8(v___x_1524_, 6, v___x_1506_);
lean_ctor_set_uint8(v___x_1524_, 7, v___x_1518_);
lean_ctor_set_uint8(v___x_1524_, 8, v___x_1506_);
lean_ctor_set_uint8(v___x_1524_, 9, v___x_1521_);
lean_ctor_set_uint8(v___x_1524_, 10, v___x_1522_);
lean_ctor_set_uint8(v___x_1524_, 11, v___x_1506_);
lean_ctor_set_uint8(v___x_1524_, 12, v___x_1506_);
lean_ctor_set_uint8(v___x_1524_, 13, v___x_1506_);
lean_ctor_set_uint8(v___x_1524_, 14, v___x_1523_);
lean_ctor_set_uint8(v___x_1524_, 15, v___x_1506_);
lean_ctor_set_uint8(v___x_1524_, 16, v___x_1506_);
lean_ctor_set_uint8(v___x_1524_, 17, v___x_1506_);
lean_ctor_set_uint8(v___x_1524_, 18, v___x_1506_);
lean_ctor_set_uint8(v___x_1524_, 19, v___x_1518_);
v___x_1525_ = l___private_Lean_Meta_Basic_0__Lean_Meta_Config_toKey(v___x_1524_);
v___x_1526_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v___x_1526_, 0, v___x_1524_);
lean_ctor_set_uint64(v___x_1526_, sizeof(void*)*1, v___x_1525_);
v___x_1527_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_higherOrderGetParam___closed__4, &lp_mathlib_Mathlib_Tactic_higherOrderGetParam___closed__4_once, _init_lp_mathlib_Mathlib_Tactic_higherOrderGetParam___closed__4);
v___x_1528_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v___x_1528_, 0, v___x_1526_);
lean_ctor_set(v___x_1528_, 1, v___x_1517_);
lean_ctor_set(v___x_1528_, 2, v___x_1527_);
lean_ctor_set(v___x_1528_, 3, v___x_1519_);
lean_ctor_set(v___x_1528_, 4, v___x_1515_);
lean_ctor_set(v___x_1528_, 5, v___x_1509_);
lean_ctor_set(v___x_1528_, 6, v___x_1515_);
lean_ctor_set_uint8(v___x_1528_, sizeof(void*)*7, v___x_1518_);
lean_ctor_set_uint8(v___x_1528_, sizeof(void*)*7 + 1, v___x_1518_);
lean_ctor_set_uint8(v___x_1528_, sizeof(void*)*7 + 2, v___x_1518_);
lean_ctor_set_uint8(v___x_1528_, sizeof(void*)*7 + 3, v___x_1506_);
v___x_1529_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_higherOrderGetParam___closed__8, &lp_mathlib_Mathlib_Tactic_higherOrderGetParam___closed__8_once, _init_lp_mathlib_Mathlib_Tactic_higherOrderGetParam___closed__8);
v___x_1530_ = lean_st_mk_ref(v___x_1529_);
v___x_1531_ = lean_box(v___x_1506_);
v___f_1532_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_higherOrderGetParam___lam__1___boxed), 11, 4);
lean_closure_set(v___f_1532_, 0, v_thm_1500_);
lean_closure_set(v___f_1532_, 1, v___x_1531_);
lean_closure_set(v___f_1532_, 2, v___y_1514_);
lean_closure_set(v___f_1532_, 3, v___y_1513_);
v___x_1533_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_higherOrderGetParam___closed__9));
v___x_1534_ = l_Lean_Elab_Term_TermElabM_run___redArg(v___f_1532_, v___x_1520_, v___x_1533_, v___x_1528_, v___x_1530_, v___y_1512_, v___y_1511_);
lean_dec_ref_known(v___x_1528_, 7);
if (lean_obj_tag(v___x_1534_) == 0)
{
lean_object* v_a_1535_; lean_object* v___x_1537_; uint8_t v_isShared_1538_; uint8_t v_isSharedCheck_1544_; 
v_a_1535_ = lean_ctor_get(v___x_1534_, 0);
v_isSharedCheck_1544_ = !lean_is_exclusive(v___x_1534_);
if (v_isSharedCheck_1544_ == 0)
{
v___x_1537_ = v___x_1534_;
v_isShared_1538_ = v_isSharedCheck_1544_;
goto v_resetjp_1536_;
}
else
{
lean_inc(v_a_1535_);
lean_dec(v___x_1534_);
v___x_1537_ = lean_box(0);
v_isShared_1538_ = v_isSharedCheck_1544_;
goto v_resetjp_1536_;
}
v_resetjp_1536_:
{
lean_object* v___x_1539_; lean_object* v_fst_1540_; lean_object* v___x_1542_; 
v___x_1539_ = lean_st_ref_get(v___x_1530_);
lean_dec(v___x_1530_);
lean_dec(v___x_1539_);
v_fst_1540_ = lean_ctor_get(v_a_1535_, 0);
lean_inc(v_fst_1540_);
lean_dec(v_a_1535_);
if (v_isShared_1538_ == 0)
{
lean_ctor_set(v___x_1537_, 0, v_fst_1540_);
v___x_1542_ = v___x_1537_;
goto v_reusejp_1541_;
}
else
{
lean_object* v_reuseFailAlloc_1543_; 
v_reuseFailAlloc_1543_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1543_, 0, v_fst_1540_);
v___x_1542_ = v_reuseFailAlloc_1543_;
goto v_reusejp_1541_;
}
v_reusejp_1541_:
{
return v___x_1542_;
}
}
}
else
{
lean_object* v_a_1545_; lean_object* v___x_1547_; uint8_t v_isShared_1548_; uint8_t v_isSharedCheck_1552_; 
lean_dec(v___x_1530_);
v_a_1545_ = lean_ctor_get(v___x_1534_, 0);
v_isSharedCheck_1552_ = !lean_is_exclusive(v___x_1534_);
if (v_isSharedCheck_1552_ == 0)
{
v___x_1547_ = v___x_1534_;
v_isShared_1548_ = v_isSharedCheck_1552_;
goto v_resetjp_1546_;
}
else
{
lean_inc(v_a_1545_);
lean_dec(v___x_1534_);
v___x_1547_ = lean_box(0);
v_isShared_1548_ = v_isSharedCheck_1552_;
goto v_resetjp_1546_;
}
v_resetjp_1546_:
{
lean_object* v___x_1550_; 
if (v_isShared_1548_ == 0)
{
v___x_1550_ = v___x_1547_;
goto v_reusejp_1549_;
}
else
{
lean_object* v_reuseFailAlloc_1551_; 
v_reuseFailAlloc_1551_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1551_, 0, v_a_1545_);
v___x_1550_ = v_reuseFailAlloc_1551_;
goto v_reusejp_1549_;
}
v_reusejp_1549_:
{
return v___x_1550_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_higherOrderGetParam___boxed(lean_object* v_thm_1565_, lean_object* v_stx_1566_, lean_object* v_a_1567_, lean_object* v_a_1568_, lean_object* v_a_1569_){
_start:
{
lean_object* v_res_1570_; 
v_res_1570_ = lp_mathlib_Mathlib_Tactic_higherOrderGetParam(v_thm_1565_, v_stx_1566_, v_a_1567_, v_a_1568_);
lean_dec(v_a_1568_);
lean_dec_ref(v_a_1567_);
return v_res_1570_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_getDeclarationRange_x3f___at___00Lean_Elab_addDeclarationRangesFromSyntax___at___00Mathlib_Tactic_higherOrderGetParam_spec__4_spec__5(lean_object* v_stx_1571_, lean_object* v___y_1572_, lean_object* v___y_1573_, lean_object* v___y_1574_, lean_object* v___y_1575_, lean_object* v___y_1576_, lean_object* v___y_1577_){
_start:
{
lean_object* v___x_1579_; 
v___x_1579_ = lp_mathlib_Lean_Elab_getDeclarationRange_x3f___at___00Lean_Elab_addDeclarationRangesFromSyntax___at___00Mathlib_Tactic_higherOrderGetParam_spec__4_spec__5___redArg(v_stx_1571_, v___y_1576_);
return v___x_1579_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_getDeclarationRange_x3f___at___00Lean_Elab_addDeclarationRangesFromSyntax___at___00Mathlib_Tactic_higherOrderGetParam_spec__4_spec__5___boxed(lean_object* v_stx_1580_, lean_object* v___y_1581_, lean_object* v___y_1582_, lean_object* v___y_1583_, lean_object* v___y_1584_, lean_object* v___y_1585_, lean_object* v___y_1586_, lean_object* v___y_1587_){
_start:
{
lean_object* v_res_1588_; 
v_res_1588_ = lp_mathlib_Lean_Elab_getDeclarationRange_x3f___at___00Lean_Elab_addDeclarationRangesFromSyntax___at___00Mathlib_Tactic_higherOrderGetParam_spec__4_spec__5(v_stx_1580_, v___y_1581_, v___y_1582_, v___y_1583_, v___y_1584_, v___y_1585_, v___y_1586_);
lean_dec(v___y_1586_);
lean_dec_ref(v___y_1585_);
lean_dec(v___y_1584_);
lean_dec_ref(v___y_1583_);
lean_dec(v___y_1582_);
lean_dec_ref(v___y_1581_);
lean_dec(v_stx_1580_);
return v_res_1588_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addDeclarationRanges___at___00Lean_Elab_addDeclarationRangesFromSyntax___at___00Mathlib_Tactic_higherOrderGetParam_spec__4_spec__6(lean_object* v_declName_1589_, lean_object* v_declRanges_1590_, lean_object* v___y_1591_, lean_object* v___y_1592_, lean_object* v___y_1593_, lean_object* v___y_1594_, lean_object* v___y_1595_, lean_object* v___y_1596_){
_start:
{
lean_object* v___x_1598_; 
v___x_1598_ = lp_mathlib_Lean_addDeclarationRanges___at___00Lean_Elab_addDeclarationRangesFromSyntax___at___00Mathlib_Tactic_higherOrderGetParam_spec__4_spec__6___redArg(v_declName_1589_, v_declRanges_1590_, v___y_1594_, v___y_1596_);
return v___x_1598_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addDeclarationRanges___at___00Lean_Elab_addDeclarationRangesFromSyntax___at___00Mathlib_Tactic_higherOrderGetParam_spec__4_spec__6___boxed(lean_object* v_declName_1599_, lean_object* v_declRanges_1600_, lean_object* v___y_1601_, lean_object* v___y_1602_, lean_object* v___y_1603_, lean_object* v___y_1604_, lean_object* v___y_1605_, lean_object* v___y_1606_, lean_object* v___y_1607_){
_start:
{
lean_object* v_res_1608_; 
v_res_1608_ = lp_mathlib_Lean_addDeclarationRanges___at___00Lean_Elab_addDeclarationRangesFromSyntax___at___00Mathlib_Tactic_higherOrderGetParam_spec__4_spec__6(v_declName_1599_, v_declRanges_1600_, v___y_1601_, v___y_1602_, v___y_1603_, v___y_1604_, v___y_1605_, v___y_1606_);
lean_dec(v___y_1606_);
lean_dec_ref(v___y_1605_);
lean_dec(v___y_1604_);
lean_dec_ref(v___y_1603_);
lean_dec(v___y_1602_);
lean_dec_ref(v___y_1601_);
return v_res_1608_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_contains___at___00Mathlib_Tactic_higherOrderGetParam_spec__6(lean_object* v_00_u03b2_1609_, lean_object* v_x_1610_, lean_object* v_x_1611_){
_start:
{
uint8_t v___x_1612_; 
v___x_1612_ = lp_mathlib_Lean_PersistentHashMap_contains___at___00Mathlib_Tactic_higherOrderGetParam_spec__6___redArg(v_x_1610_, v_x_1611_);
return v___x_1612_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_contains___at___00Mathlib_Tactic_higherOrderGetParam_spec__6___boxed(lean_object* v_00_u03b2_1613_, lean_object* v_x_1614_, lean_object* v_x_1615_){
_start:
{
uint8_t v_res_1616_; lean_object* v_r_1617_; 
v_res_1616_ = lp_mathlib_Lean_PersistentHashMap_contains___at___00Mathlib_Tactic_higherOrderGetParam_spec__6(v_00_u03b2_1613_, v_x_1614_, v_x_1615_);
lean_dec_ref(v_x_1615_);
lean_dec_ref(v_x_1614_);
v_r_1617_ = lean_box(v_res_1616_);
return v_r_1617_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_higherOrderGetParam_spec__7(lean_object* v_00_u03b1_1618_, lean_object* v_msg_1619_, lean_object* v___y_1620_, lean_object* v___y_1621_, lean_object* v___y_1622_, lean_object* v___y_1623_, lean_object* v___y_1624_, lean_object* v___y_1625_){
_start:
{
lean_object* v___x_1627_; 
v___x_1627_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_higherOrderGetParam_spec__7___redArg(v_msg_1619_, v___y_1620_, v___y_1621_, v___y_1622_, v___y_1623_, v___y_1624_, v___y_1625_);
return v___x_1627_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_higherOrderGetParam_spec__7___boxed(lean_object* v_00_u03b1_1628_, lean_object* v_msg_1629_, lean_object* v___y_1630_, lean_object* v___y_1631_, lean_object* v___y_1632_, lean_object* v___y_1633_, lean_object* v___y_1634_, lean_object* v___y_1635_, lean_object* v___y_1636_){
_start:
{
lean_object* v_res_1637_; 
v_res_1637_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_higherOrderGetParam_spec__7(v_00_u03b1_1628_, v_msg_1629_, v___y_1630_, v___y_1631_, v___y_1632_, v___y_1633_, v___y_1634_, v___y_1635_);
lean_dec(v___y_1635_);
lean_dec_ref(v___y_1634_);
lean_dec(v___y_1633_);
lean_dec_ref(v___y_1632_);
lean_dec(v___y_1631_);
lean_dec_ref(v___y_1630_);
return v_res_1637_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forM___at___00Mathlib_Tactic_higherOrderGetParam_spec__8(lean_object* v_as_1638_, lean_object* v___y_1639_, lean_object* v___y_1640_, lean_object* v___y_1641_, lean_object* v___y_1642_, lean_object* v___y_1643_, lean_object* v___y_1644_){
_start:
{
lean_object* v___x_1646_; 
v___x_1646_ = lp_mathlib_List_forM___at___00Mathlib_Tactic_higherOrderGetParam_spec__8___redArg(v_as_1638_, v___y_1641_, v___y_1642_, v___y_1643_, v___y_1644_);
return v___x_1646_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forM___at___00Mathlib_Tactic_higherOrderGetParam_spec__8___boxed(lean_object* v_as_1647_, lean_object* v___y_1648_, lean_object* v___y_1649_, lean_object* v___y_1650_, lean_object* v___y_1651_, lean_object* v___y_1652_, lean_object* v___y_1653_, lean_object* v___y_1654_){
_start:
{
lean_object* v_res_1655_; 
v_res_1655_ = lp_mathlib_List_forM___at___00Mathlib_Tactic_higherOrderGetParam_spec__8(v_as_1647_, v___y_1648_, v___y_1649_, v___y_1650_, v___y_1651_, v___y_1652_, v___y_1653_);
lean_dec(v___y_1653_);
lean_dec_ref(v___y_1652_);
lean_dec(v___y_1651_);
lean_dec_ref(v___y_1650_);
lean_dec(v___y_1649_);
lean_dec_ref(v___y_1648_);
return v_res_1655_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1(lean_object* v_00_u03b1_1656_, lean_object* v_constName_1657_, lean_object* v___y_1658_, lean_object* v___y_1659_, lean_object* v___y_1660_, lean_object* v___y_1661_, lean_object* v___y_1662_, lean_object* v___y_1663_){
_start:
{
lean_object* v___x_1665_; 
v___x_1665_ = lp_mathlib_Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1___redArg(v_constName_1657_, v___y_1658_, v___y_1659_, v___y_1660_, v___y_1661_, v___y_1662_, v___y_1663_);
return v___x_1665_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1___boxed(lean_object* v_00_u03b1_1666_, lean_object* v_constName_1667_, lean_object* v___y_1668_, lean_object* v___y_1669_, lean_object* v___y_1670_, lean_object* v___y_1671_, lean_object* v___y_1672_, lean_object* v___y_1673_, lean_object* v___y_1674_){
_start:
{
lean_object* v_res_1675_; 
v_res_1675_ = lp_mathlib_Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1(v_00_u03b1_1666_, v_constName_1667_, v___y_1668_, v___y_1669_, v___y_1670_, v___y_1671_, v___y_1672_, v___y_1673_);
lean_dec(v___y_1673_);
lean_dec_ref(v___y_1672_);
lean_dec(v___y_1671_);
lean_dec_ref(v___y_1670_);
lean_dec(v___y_1669_);
lean_dec_ref(v___y_1668_);
return v_res_1675_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Mathlib_Tactic_higherOrderGetParam_spec__6_spec__10(lean_object* v_00_u03b2_1676_, lean_object* v_x_1677_, size_t v_x_1678_, lean_object* v_x_1679_){
_start:
{
uint8_t v___x_1680_; 
v___x_1680_ = lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Mathlib_Tactic_higherOrderGetParam_spec__6_spec__10___redArg(v_x_1677_, v_x_1678_, v_x_1679_);
return v___x_1680_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Mathlib_Tactic_higherOrderGetParam_spec__6_spec__10___boxed(lean_object* v_00_u03b2_1681_, lean_object* v_x_1682_, lean_object* v_x_1683_, lean_object* v_x_1684_){
_start:
{
size_t v_x_20250__boxed_1685_; uint8_t v_res_1686_; lean_object* v_r_1687_; 
v_x_20250__boxed_1685_ = lean_unbox_usize(v_x_1683_);
lean_dec(v_x_1683_);
v_res_1686_ = lp_mathlib_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Mathlib_Tactic_higherOrderGetParam_spec__6_spec__10(v_00_u03b2_1681_, v_x_1682_, v_x_20250__boxed_1685_, v_x_1684_);
lean_dec_ref(v_x_1684_);
lean_dec_ref(v_x_1682_);
v_r_1687_ = lean_box(v_res_1686_);
return v_r_1687_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_higherOrderGetParam_spec__7_spec__12(lean_object* v_msgData_1688_, lean_object* v_macroStack_1689_, lean_object* v___y_1690_, lean_object* v___y_1691_, lean_object* v___y_1692_, lean_object* v___y_1693_, lean_object* v___y_1694_, lean_object* v___y_1695_){
_start:
{
lean_object* v___x_1697_; 
v___x_1697_ = lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_higherOrderGetParam_spec__7_spec__12___redArg(v_msgData_1688_, v_macroStack_1689_, v___y_1694_);
return v___x_1697_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_higherOrderGetParam_spec__7_spec__12___boxed(lean_object* v_msgData_1698_, lean_object* v_macroStack_1699_, lean_object* v___y_1700_, lean_object* v___y_1701_, lean_object* v___y_1702_, lean_object* v___y_1703_, lean_object* v___y_1704_, lean_object* v___y_1705_, lean_object* v___y_1706_){
_start:
{
lean_object* v_res_1707_; 
v_res_1707_ = lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Mathlib_Tactic_higherOrderGetParam_spec__7_spec__12(v_msgData_1698_, v_macroStack_1699_, v___y_1700_, v___y_1701_, v___y_1702_, v___y_1703_, v___y_1704_, v___y_1705_);
lean_dec(v___y_1705_);
lean_dec_ref(v___y_1704_);
lean_dec(v___y_1703_);
lean_dec_ref(v___y_1702_);
lean_dec(v___y_1701_);
lean_dec_ref(v___y_1700_);
return v_res_1707_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3(lean_object* v_00_u03b1_1708_, lean_object* v_ref_1709_, lean_object* v_constName_1710_, lean_object* v___y_1711_, lean_object* v___y_1712_, lean_object* v___y_1713_, lean_object* v___y_1714_, lean_object* v___y_1715_, lean_object* v___y_1716_){
_start:
{
lean_object* v___x_1718_; 
v___x_1718_ = lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3___redArg(v_ref_1709_, v_constName_1710_, v___y_1711_, v___y_1712_, v___y_1713_, v___y_1714_, v___y_1715_, v___y_1716_);
return v___x_1718_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3___boxed(lean_object* v_00_u03b1_1719_, lean_object* v_ref_1720_, lean_object* v_constName_1721_, lean_object* v___y_1722_, lean_object* v___y_1723_, lean_object* v___y_1724_, lean_object* v___y_1725_, lean_object* v___y_1726_, lean_object* v___y_1727_, lean_object* v___y_1728_){
_start:
{
lean_object* v_res_1729_; 
v_res_1729_ = lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3(v_00_u03b1_1719_, v_ref_1720_, v_constName_1721_, v___y_1722_, v___y_1723_, v___y_1724_, v___y_1725_, v___y_1726_, v___y_1727_);
lean_dec(v___y_1727_);
lean_dec_ref(v___y_1726_);
lean_dec(v___y_1725_);
lean_dec_ref(v___y_1724_);
lean_dec(v___y_1723_);
lean_dec_ref(v___y_1722_);
lean_dec(v_ref_1720_);
return v_res_1729_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Mathlib_Tactic_higherOrderGetParam_spec__6_spec__10_spec__12(lean_object* v_00_u03b2_1730_, lean_object* v_keys_1731_, lean_object* v_vals_1732_, lean_object* v_heq_1733_, lean_object* v_i_1734_, lean_object* v_k_1735_){
_start:
{
uint8_t v___x_1736_; 
v___x_1736_ = lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Mathlib_Tactic_higherOrderGetParam_spec__6_spec__10_spec__12___redArg(v_keys_1731_, v_i_1734_, v_k_1735_);
return v___x_1736_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Mathlib_Tactic_higherOrderGetParam_spec__6_spec__10_spec__12___boxed(lean_object* v_00_u03b2_1737_, lean_object* v_keys_1738_, lean_object* v_vals_1739_, lean_object* v_heq_1740_, lean_object* v_i_1741_, lean_object* v_k_1742_){
_start:
{
uint8_t v_res_1743_; lean_object* v_r_1744_; 
v_res_1743_ = lp_mathlib_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Mathlib_Tactic_higherOrderGetParam_spec__6_spec__10_spec__12(v_00_u03b2_1737_, v_keys_1738_, v_vals_1739_, v_heq_1740_, v_i_1741_, v_k_1742_);
lean_dec_ref(v_k_1742_);
lean_dec_ref(v_vals_1739_);
lean_dec_ref(v_keys_1738_);
v_r_1744_ = lean_box(v_res_1743_);
return v_r_1744_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3_spec__12(lean_object* v_00_u03b1_1745_, lean_object* v_ref_1746_, lean_object* v_msg_1747_, lean_object* v_declHint_1748_, lean_object* v___y_1749_, lean_object* v___y_1750_, lean_object* v___y_1751_, lean_object* v___y_1752_, lean_object* v___y_1753_, lean_object* v___y_1754_){
_start:
{
lean_object* v___x_1756_; 
v___x_1756_ = lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3_spec__12___redArg(v_ref_1746_, v_msg_1747_, v_declHint_1748_, v___y_1749_, v___y_1750_, v___y_1751_, v___y_1752_, v___y_1753_, v___y_1754_);
return v___x_1756_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3_spec__12___boxed(lean_object* v_00_u03b1_1757_, lean_object* v_ref_1758_, lean_object* v_msg_1759_, lean_object* v_declHint_1760_, lean_object* v___y_1761_, lean_object* v___y_1762_, lean_object* v___y_1763_, lean_object* v___y_1764_, lean_object* v___y_1765_, lean_object* v___y_1766_, lean_object* v___y_1767_){
_start:
{
lean_object* v_res_1768_; 
v_res_1768_ = lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3_spec__12(v_00_u03b1_1757_, v_ref_1758_, v_msg_1759_, v_declHint_1760_, v___y_1761_, v___y_1762_, v___y_1763_, v___y_1764_, v___y_1765_, v___y_1766_);
lean_dec(v___y_1766_);
lean_dec_ref(v___y_1765_);
lean_dec(v___y_1764_);
lean_dec_ref(v___y_1763_);
lean_dec(v___y_1762_);
lean_dec_ref(v___y_1761_);
lean_dec(v_ref_1758_);
return v_res_1768_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3_spec__12_spec__17_spec__20(lean_object* v_msg_1769_, lean_object* v_declHint_1770_, lean_object* v___y_1771_, lean_object* v___y_1772_, lean_object* v___y_1773_, lean_object* v___y_1774_, lean_object* v___y_1775_, lean_object* v___y_1776_){
_start:
{
lean_object* v___x_1778_; 
v___x_1778_ = lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3_spec__12_spec__17_spec__20___redArg(v_msg_1769_, v_declHint_1770_, v___y_1776_);
return v___x_1778_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3_spec__12_spec__17_spec__20___boxed(lean_object* v_msg_1779_, lean_object* v_declHint_1780_, lean_object* v___y_1781_, lean_object* v___y_1782_, lean_object* v___y_1783_, lean_object* v___y_1784_, lean_object* v___y_1785_, lean_object* v___y_1786_, lean_object* v___y_1787_){
_start:
{
lean_object* v_res_1788_; 
v_res_1788_ = lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3_spec__12_spec__17_spec__20(v_msg_1779_, v_declHint_1780_, v___y_1781_, v___y_1782_, v___y_1783_, v___y_1784_, v___y_1785_, v___y_1786_);
lean_dec(v___y_1786_);
lean_dec_ref(v___y_1785_);
lean_dec(v___y_1784_);
lean_dec_ref(v___y_1783_);
lean_dec(v___y_1782_);
lean_dec_ref(v___y_1781_);
return v_res_1788_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3_spec__12_spec__18(lean_object* v_00_u03b1_1789_, lean_object* v_ref_1790_, lean_object* v_msg_1791_, lean_object* v___y_1792_, lean_object* v___y_1793_, lean_object* v___y_1794_, lean_object* v___y_1795_, lean_object* v___y_1796_, lean_object* v___y_1797_){
_start:
{
lean_object* v___x_1799_; 
v___x_1799_ = lp_mathlib_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3_spec__12_spec__18___redArg(v_ref_1790_, v_msg_1791_, v___y_1792_, v___y_1793_, v___y_1794_, v___y_1795_, v___y_1796_, v___y_1797_);
return v___x_1799_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3_spec__12_spec__18___boxed(lean_object* v_00_u03b1_1800_, lean_object* v_ref_1801_, lean_object* v_msg_1802_, lean_object* v___y_1803_, lean_object* v___y_1804_, lean_object* v___y_1805_, lean_object* v___y_1806_, lean_object* v___y_1807_, lean_object* v___y_1808_, lean_object* v___y_1809_){
_start:
{
lean_object* v_res_1810_; 
v_res_1810_ = lp_mathlib_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00Mathlib_Tactic_higherOrderGetParam_spec__1_spec__1_spec__3_spec__12_spec__18(v_00_u03b1_1800_, v_ref_1801_, v_msg_1802_, v___y_1803_, v___y_1804_, v___y_1805_, v___y_1806_, v___y_1807_, v___y_1808_);
lean_dec(v___y_1808_);
lean_dec_ref(v___y_1807_);
lean_dec(v___y_1806_);
lean_dec_ref(v___y_1805_);
lean_dec(v___y_1804_);
lean_dec_ref(v___y_1803_);
lean_dec(v_ref_1801_);
return v_res_1810_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_HigherOrder_0__Mathlib_Tactic_initFn___lam__0_00___x40_Mathlib_Tactic_HigherOrder_2684385311____hygCtx___hyg_2_(lean_object* v_x_1811_, lean_object* v_x_1812_, lean_object* v_x_1813_, lean_object* v___y_1814_){
_start:
{
lean_object* v___x_1816_; lean_object* v___x_1817_; 
v___x_1816_ = lean_box(0);
v___x_1817_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1817_, 0, v___x_1816_);
return v___x_1817_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_HigherOrder_0__Mathlib_Tactic_initFn___lam__0_00___x40_Mathlib_Tactic_HigherOrder_2684385311____hygCtx___hyg_2____boxed(lean_object* v_x_1818_, lean_object* v_x_1819_, lean_object* v_x_1820_, lean_object* v___y_1821_, lean_object* v___y_1822_){
_start:
{
lean_object* v_res_1823_; 
v_res_1823_ = lp_mathlib___private_Mathlib_Tactic_HigherOrder_0__Mathlib_Tactic_initFn___lam__0_00___x40_Mathlib_Tactic_HigherOrder_2684385311____hygCtx___hyg_2_(v_x_1818_, v_x_1819_, v_x_1820_, v___y_1821_);
lean_dec(v___y_1821_);
lean_dec_ref(v_x_1820_);
lean_dec(v_x_1819_);
lean_dec(v_x_1818_);
return v_res_1823_;
}
}
LEAN_EXPORT uint8_t lp_mathlib___private_Mathlib_Tactic_HigherOrder_0__Mathlib_Tactic_initFn___lam__1_00___x40_Mathlib_Tactic_HigherOrder_2684385311____hygCtx___hyg_2_(uint8_t v___x_1824_, lean_object* v_env_1825_, lean_object* v_n_1826_, lean_object* v_x_1827_){
_start:
{
uint8_t v___x_1828_; 
v___x_1828_ = l_Lean_Environment_contains(v_env_1825_, v_n_1826_, v___x_1824_);
return v___x_1828_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_HigherOrder_0__Mathlib_Tactic_initFn___lam__1_00___x40_Mathlib_Tactic_HigherOrder_2684385311____hygCtx___hyg_2____boxed(lean_object* v___x_1829_, lean_object* v_env_1830_, lean_object* v_n_1831_, lean_object* v_x_1832_){
_start:
{
uint8_t v___x_99__boxed_1833_; uint8_t v_res_1834_; lean_object* v_r_1835_; 
v___x_99__boxed_1833_ = lean_unbox(v___x_1829_);
v_res_1834_ = lp_mathlib___private_Mathlib_Tactic_HigherOrder_0__Mathlib_Tactic_initFn___lam__1_00___x40_Mathlib_Tactic_HigherOrder_2684385311____hygCtx___hyg_2_(v___x_99__boxed_1833_, v_env_1830_, v_n_1831_, v_x_1832_);
lean_dec(v_x_1832_);
v_r_1835_ = lean_box(v_res_1834_);
return v_r_1835_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_HigherOrder_0__Mathlib_Tactic_initFn_00___x40_Mathlib_Tactic_HigherOrder_2684385311____hygCtx___hyg_2_(){
_start:
{
lean_object* v___x_1863_; lean_object* v___x_1864_; 
v___x_1863_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_HigherOrder_0__Mathlib_Tactic_initFn___closed__10_00___x40_Mathlib_Tactic_HigherOrder_2684385311____hygCtx___hyg_2_));
v___x_1864_ = l_Lean_registerParametricAttribute___redArg(v___x_1863_);
return v___x_1864_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_HigherOrder_0__Mathlib_Tactic_initFn_00___x40_Mathlib_Tactic_HigherOrder_2684385311____hygCtx___hyg_2____boxed(lean_object* v_a_1865_){
_start:
{
lean_object* v_res_1866_; 
v_res_1866_ = lp_mathlib___private_Mathlib_Tactic_HigherOrder_0__Mathlib_Tactic_initFn_00___x40_Mathlib_Tactic_HigherOrder_2684385311____hygCtx___hyg_2_();
return v_res_1866_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_Lean_Meta_Tactic_Simp(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Init(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Tactic_HigherOrder(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Meta_Tactic_Simp(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_Lean_Meta_Tactic_Apply(uint8_t builtin);
lean_object* runtime_initialize_Lean_Meta_Tactic_Assumption(uint8_t builtin);
lean_object* runtime_initialize_Lean_Meta_MatchUtil(uint8_t builtin);
lean_object* runtime_initialize_Lean_Meta_Tactic_Intro(uint8_t builtin);
lean_object* runtime_initialize_Lean_Elab_DeclarationRange(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Tactic_HigherOrder(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Meta_Tactic_Apply(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Meta_Tactic_Assumption(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Meta_MatchUtil(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Meta_Tactic_Intro(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Elab_DeclarationRange(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = lp_mathlib___private_Mathlib_Tactic_HigherOrder_0__Mathlib_Tactic_initFn_00___x40_Mathlib_Tactic_HigherOrder_2684385311____hygCtx___hyg_2_();
if (lean_io_result_is_error(res)) return res;
lp_mathlib_Mathlib_Tactic_higherOrderAttr = lean_io_result_get_value(res);
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_higherOrderAttr);
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Lean_Meta_Tactic_Apply(uint8_t builtin);
lean_object* initialize_Lean_Meta_Tactic_Assumption(uint8_t builtin);
lean_object* initialize_Lean_Meta_MatchUtil(uint8_t builtin);
lean_object* initialize_Lean_Meta_Tactic_Intro(uint8_t builtin);
lean_object* initialize_Lean_Elab_DeclarationRange(uint8_t builtin);
lean_object* initialize_Lean_Meta_Tactic_Simp(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Init(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_HigherOrder(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Meta_Tactic_Apply(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Meta_Tactic_Assumption(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Meta_MatchUtil(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Meta_Tactic_Intro(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Elab_DeclarationRange(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Meta_Tactic_Simp(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_HigherOrder(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Tactic_HigherOrder(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Tactic_HigherOrder(builtin);
}
#ifdef __cplusplus
}
#endif
