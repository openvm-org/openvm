// Lean compiler output
// Module: Aesop.RuleTac.Tactic
// Imports: public import Init public meta import Init public import Aesop.RuleTac.Basic
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
lean_object* lp_aesop_Aesop_Script_Tactic_unstructured(lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
size_t lean_array_size(lean_object*);
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
lean_object* lp_aesop_Aesop_mvarIdToSubgoal(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
size_t lean_usize_add(size_t, size_t);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* lean_st_ref_get(lean_object*);
uint8_t lean_has_compile_error(lean_object*, lean_object*);
lean_object* l_Lean_Environment_evalConst___redArg(lean_object*, lean_object*, lean_object*, uint8_t);
extern lean_object* l_Lean_Elab_abortCommandExceptionId;
lean_object* l_Lean_MetavarContext_getExprAssignmentCore_x3f(lean_object*, lean_object*);
lean_object* lean_array_uget(lean_object*, size_t);
lean_object* l_Lean_Meta_SavedState_restore___redArg(lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Exception_isInterrupt(lean_object*);
uint8_t l_Lean_Exception_isRuntime(lean_object*);
lean_object* lp_aesop_Aesop_Percent_ofFloat(double);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Lean_Parser_runParserCategory(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_evalTactic___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_run___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Term_TermElabM_run___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_saveState___redArg(lean_object*, lean_object*);
lean_object* lean_array_mk(lean_object*);
uint8_t lp_aesop___private_Aesop_Util_Basic_0__Aesop_hasSorry_go(lean_object*, lean_object*);
lean_object* l_Lean_MessageData_ofFormat(lean_object*);
lean_object* l_Lean_indentD(lean_object*);
lean_object* lean_float_to_string(double);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_withoutRecover___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Syntax_node3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_List_reverse___redArg(lean_object*);
lean_object* l_Lean_Exception_toMessageData(lean_object*);
lean_object* lean_array_get_size(lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* lean_array_to_list(lean_object*);
lean_object* l_Lean_MessageData_joinSep(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_RuleTac_tacticMImpl___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTac_tacticMImpl___lam__0___boxed(lean_object*);
static lean_once_cell_t lp_aesop_Lean_Elab_throwAbortCommand___at___00Lean_evalConst___at___00Aesop_RuleTac_tacticMImpl_spec__0_spec__1___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Lean_Elab_throwAbortCommand___at___00Lean_evalConst___at___00Aesop_RuleTac_tacticMImpl_spec__0_spec__1___redArg___closed__0;
LEAN_EXPORT lean_object* lp_aesop_Lean_Elab_throwAbortCommand___at___00Lean_evalConst___at___00Aesop_RuleTac_tacticMImpl_spec__0_spec__1___redArg();
LEAN_EXPORT lean_object* lp_aesop_Lean_Elab_throwAbortCommand___at___00Lean_evalConst___at___00Aesop_RuleTac_tacticMImpl_spec__0_spec__1___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_ofExcept___at___00Lean_evalConst___at___00Aesop_RuleTac_tacticMImpl_spec__0_spec__0_spec__1_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_ofExcept___at___00Lean_evalConst___at___00Aesop_RuleTac_tacticMImpl_spec__0_spec__0_spec__1_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Lean_ofExcept___at___00Lean_evalConst___at___00Aesop_RuleTac_tacticMImpl_spec__0_spec__0_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Lean_ofExcept___at___00Lean_evalConst___at___00Aesop_RuleTac_tacticMImpl_spec__0_spec__0_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_ofExcept___at___00Lean_evalConst___at___00Aesop_RuleTac_tacticMImpl_spec__0_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_ofExcept___at___00Lean_evalConst___at___00Aesop_RuleTac_tacticMImpl_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_evalConst___at___00Aesop_RuleTac_tacticMImpl_spec__0___redArg(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_evalConst___at___00Aesop_RuleTac_tacticMImpl_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_List_mapM_loop___at___00Aesop_RuleTac_tacticMImpl_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_List_mapM_loop___at___00Aesop_RuleTac_tacticMImpl_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_aesop_Aesop_RuleTac_tacticMImpl___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_RuleTac_tacticMImpl___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_RuleTac_tacticMImpl___closed__0 = (const lean_object*)&lp_aesop_Aesop_RuleTac_tacticMImpl___closed__0_value;
static const lean_array_object lp_aesop_Aesop_RuleTac_tacticMImpl___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_aesop_Aesop_RuleTac_tacticMImpl___closed__1 = (const lean_object*)&lp_aesop_Aesop_RuleTac_tacticMImpl___closed__1_value;
static const lean_ctor_object lp_aesop_Aesop_RuleTac_tacticMImpl___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*8 + 16, .m_other = 8, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_RuleTac_tacticMImpl___closed__0_value),((lean_object*)(((size_t)(1) << 1) | 1)),((lean_object*)(((size_t)(1) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_RuleTac_tacticMImpl___closed__1_value),LEAN_SCALAR_PTR_LITERAL(1, 1, 1, 1, 0, 0, 0, 0),LEAN_SCALAR_PTR_LITERAL(1, 0, 1, 0, 0, 0, 0, 0)}};
static const lean_object* lp_aesop_Aesop_RuleTac_tacticMImpl___closed__2 = (const lean_object*)&lp_aesop_Aesop_RuleTac_tacticMImpl___closed__2_value;
static const lean_ctor_object lp_aesop_Aesop_RuleTac_tacticMImpl___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*7 + 0, .m_other = 7, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(1) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(1) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_aesop_Aesop_RuleTac_tacticMImpl___closed__3 = (const lean_object*)&lp_aesop_Aesop_RuleTac_tacticMImpl___closed__3_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTac_tacticMImpl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTac_tacticMImpl___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Elab_throwAbortCommand___at___00Lean_evalConst___at___00Aesop_RuleTac_tacticMImpl_spec__0_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Elab_throwAbortCommand___at___00Lean_evalConst___at___00Aesop_RuleTac_tacticMImpl_spec__0_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_evalConst___at___00Aesop_RuleTac_tacticMImpl_spec__0(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_evalConst___at___00Aesop_RuleTac_tacticMImpl_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_ofExcept___at___00Lean_evalConst___at___00Aesop_RuleTac_tacticMImpl_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_ofExcept___at___00Lean_evalConst___at___00Aesop_RuleTac_tacticMImpl_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Lean_ofExcept___at___00Lean_evalConst___at___00Aesop_RuleTac_tacticMImpl_spec__0_spec__0_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Lean_ofExcept___at___00Lean_evalConst___at___00Aesop_RuleTac_tacticMImpl_spec__0_spec__0_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTac_ruleTacImpl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTac_ruleTacImpl___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTac_singleRuleTacImpl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTac_singleRuleTacImpl___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_RuleTac_tacticStx___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTac_tacticStx___lam__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_RuleTac_tacticStx_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_RuleTac_tacticStx_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop_Aesop_RuleTac_tacticStx___lam__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_aesop_Aesop_RuleTac_tacticStx___lam__1___closed__0 = (const lean_object*)&lp_aesop_Aesop_RuleTac_tacticStx___lam__1___closed__0_value;
static const lean_string_object lp_aesop_Aesop_RuleTac_tacticStx___lam__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_aesop_Aesop_RuleTac_tacticStx___lam__1___closed__1 = (const lean_object*)&lp_aesop_Aesop_RuleTac_tacticStx___lam__1___closed__1_value;
static const lean_string_object lp_aesop_Aesop_RuleTac_tacticStx___lam__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_aesop_Aesop_RuleTac_tacticStx___lam__1___closed__2 = (const lean_object*)&lp_aesop_Aesop_RuleTac_tacticStx___lam__1___closed__2_value;
static const lean_string_object lp_aesop_Aesop_RuleTac_tacticStx___lam__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "tacticSeq"};
static const lean_object* lp_aesop_Aesop_RuleTac_tacticStx___lam__1___closed__3 = (const lean_object*)&lp_aesop_Aesop_RuleTac_tacticStx___lam__1___closed__3_value;
static const lean_ctor_object lp_aesop_Aesop_RuleTac_tacticStx___lam__1___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_RuleTac_tacticStx___lam__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_aesop_Aesop_RuleTac_tacticStx___lam__1___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_RuleTac_tacticStx___lam__1___closed__4_value_aux_0),((lean_object*)&lp_aesop_Aesop_RuleTac_tacticStx___lam__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_aesop_Aesop_RuleTac_tacticStx___lam__1___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_RuleTac_tacticStx___lam__1___closed__4_value_aux_1),((lean_object*)&lp_aesop_Aesop_RuleTac_tacticStx___lam__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_aesop_Aesop_RuleTac_tacticStx___lam__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_RuleTac_tacticStx___lam__1___closed__4_value_aux_2),((lean_object*)&lp_aesop_Aesop_RuleTac_tacticStx___lam__1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(212, 140, 85, 215, 241, 69, 7, 118)}};
static const lean_object* lp_aesop_Aesop_RuleTac_tacticStx___lam__1___closed__4 = (const lean_object*)&lp_aesop_Aesop_RuleTac_tacticStx___lam__1___closed__4_value;
static const lean_string_object lp_aesop_Aesop_RuleTac_tacticStx___lam__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 57, .m_capacity = 57, .m_length = 56, .m_data = "expected either a single tactic or a sequence of tactics"};
static const lean_object* lp_aesop_Aesop_RuleTac_tacticStx___lam__1___closed__5 = (const lean_object*)&lp_aesop_Aesop_RuleTac_tacticStx___lam__1___closed__5_value;
static lean_once_cell_t lp_aesop_Aesop_RuleTac_tacticStx___lam__1___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_RuleTac_tacticStx___lam__1___closed__6;
static const lean_string_object lp_aesop_Aesop_RuleTac_tacticStx___lam__1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "paren"};
static const lean_object* lp_aesop_Aesop_RuleTac_tacticStx___lam__1___closed__7 = (const lean_object*)&lp_aesop_Aesop_RuleTac_tacticStx___lam__1___closed__7_value;
static const lean_ctor_object lp_aesop_Aesop_RuleTac_tacticStx___lam__1___closed__8_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_RuleTac_tacticStx___lam__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_aesop_Aesop_RuleTac_tacticStx___lam__1___closed__8_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_RuleTac_tacticStx___lam__1___closed__8_value_aux_0),((lean_object*)&lp_aesop_Aesop_RuleTac_tacticStx___lam__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_aesop_Aesop_RuleTac_tacticStx___lam__1___closed__8_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_RuleTac_tacticStx___lam__1___closed__8_value_aux_1),((lean_object*)&lp_aesop_Aesop_RuleTac_tacticStx___lam__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_aesop_Aesop_RuleTac_tacticStx___lam__1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_RuleTac_tacticStx___lam__1___closed__8_value_aux_2),((lean_object*)&lp_aesop_Aesop_RuleTac_tacticStx___lam__1___closed__7_value),LEAN_SCALAR_PTR_LITERAL(117, 253, 122, 28, 77, 248, 149, 120)}};
static const lean_object* lp_aesop_Aesop_RuleTac_tacticStx___lam__1___closed__8 = (const lean_object*)&lp_aesop_Aesop_RuleTac_tacticStx___lam__1___closed__8_value;
static const lean_string_object lp_aesop_Aesop_RuleTac_tacticStx___lam__1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "("};
static const lean_object* lp_aesop_Aesop_RuleTac_tacticStx___lam__1___closed__9 = (const lean_object*)&lp_aesop_Aesop_RuleTac_tacticStx___lam__1___closed__9_value;
static const lean_string_object lp_aesop_Aesop_RuleTac_tacticStx___lam__1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ")"};
static const lean_object* lp_aesop_Aesop_RuleTac_tacticStx___lam__1___closed__10 = (const lean_object*)&lp_aesop_Aesop_RuleTac_tacticStx___lam__1___closed__10_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTac_tacticStx___lam__1(uint8_t, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTac_tacticStx___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_RuleTac_tacticStx_spec__1(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_RuleTac_tacticStx_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_aesop_Aesop_RuleTac_tacticStx___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_RuleTac_tacticStx___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_RuleTac_tacticStx___closed__0 = (const lean_object*)&lp_aesop_Aesop_RuleTac_tacticStx___closed__0_value;
static const lean_ctor_object lp_aesop_Aesop_RuleTac_tacticStx___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*8 + 16, .m_other = 8, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_RuleTac_tacticStx___closed__0_value),((lean_object*)(((size_t)(1) << 1) | 1)),((lean_object*)(((size_t)(1) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_RuleTac_tacticMImpl___closed__1_value),LEAN_SCALAR_PTR_LITERAL(1, 1, 1, 1, 0, 0, 0, 0),LEAN_SCALAR_PTR_LITERAL(1, 0, 1, 0, 0, 0, 0, 0)}};
static const lean_object* lp_aesop_Aesop_RuleTac_tacticStx___closed__1 = (const lean_object*)&lp_aesop_Aesop_RuleTac_tacticStx___closed__1_value;
static const lean_string_object lp_aesop_Aesop_RuleTac_tacticStx___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "tactic"};
static const lean_object* lp_aesop_Aesop_RuleTac_tacticStx___closed__2 = (const lean_object*)&lp_aesop_Aesop_RuleTac_tacticStx___closed__2_value;
static const lean_ctor_object lp_aesop_Aesop_RuleTac_tacticStx___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_RuleTac_tacticStx___closed__2_value),LEAN_SCALAR_PTR_LITERAL(99, 76, 33, 121, 85, 143, 17, 224)}};
static const lean_object* lp_aesop_Aesop_RuleTac_tacticStx___closed__3 = (const lean_object*)&lp_aesop_Aesop_RuleTac_tacticStx___closed__3_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTac_tacticStx(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTac_tacticStx___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_RuleTac_tacticStx_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_RuleTac_tacticStx_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_getExprMVarAssignment_x3f___at___00Aesop_RuleTac_tacGenImpl_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_getExprMVarAssignment_x3f___at___00Aesop_RuleTac_tacGenImpl_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_getExprMVarAssignment_x3f___at___00Aesop_RuleTac_tacGenImpl_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_getExprMVarAssignment_x3f___at___00Aesop_RuleTac_tacGenImpl_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_hasSorry___at___00Aesop_RuleTac_tacGenImpl_spec__1___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_hasSorry___at___00Aesop_RuleTac_tacGenImpl_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_hasSorry___at___00Aesop_RuleTac_tacGenImpl_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_hasSorry___at___00Aesop_RuleTac_tacGenImpl_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RuleTac_tacGenImpl_spec__2___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RuleTac_tacGenImpl_spec__2___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RuleTac_tacGenImpl_spec__2___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RuleTac_tacGenImpl_spec__2___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RuleTac_tacGenImpl_spec__2___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "<stdin>"};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RuleTac_tacGenImpl_spec__2___closed__0 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RuleTac_tacGenImpl_spec__2___closed__0_value;
static const lean_string_object lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RuleTac_tacGenImpl_spec__2___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 31, .m_capacity = 31, .m_length = 30, .m_data = "generated proof contains sorry"};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RuleTac_tacGenImpl_spec__2___closed__1 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RuleTac_tacGenImpl_spec__2___closed__1_value;
static lean_once_cell_t lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RuleTac_tacGenImpl_spec__2___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RuleTac_tacGenImpl_spec__2___closed__2;
static const lean_string_object lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RuleTac_tacGenImpl_spec__2___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 30, .m_capacity = 30, .m_length = 29, .m_data = "failed to parse tactic syntax"};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RuleTac_tacGenImpl_spec__2___closed__3 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RuleTac_tacGenImpl_spec__2___closed__3_value;
static lean_once_cell_t lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RuleTac_tacGenImpl_spec__2___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RuleTac_tacGenImpl_spec__2___closed__4;
static const lean_string_object lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RuleTac_tacGenImpl_spec__2___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 30, .m_capacity = 30, .m_length = 29, .m_data = "invalid success probability '"};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RuleTac_tacGenImpl_spec__2___closed__5 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RuleTac_tacGenImpl_spec__2___closed__5_value;
static lean_once_cell_t lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RuleTac_tacGenImpl_spec__2___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RuleTac_tacGenImpl_spec__2___closed__6;
static const lean_string_object lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RuleTac_tacGenImpl_spec__2___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 27, .m_capacity = 27, .m_length = 26, .m_data = "', must be between 0 and 1"};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RuleTac_tacGenImpl_spec__2___closed__7 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RuleTac_tacGenImpl_spec__2___closed__7_value;
static lean_once_cell_t lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RuleTac_tacGenImpl_spec__2___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RuleTac_tacGenImpl_spec__2___closed__8;
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RuleTac_tacGenImpl_spec__2(lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RuleTac_tacGenImpl_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop_List_mapTR_loop___at___00Aesop_RuleTac_tacGenImpl_spec__3___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = ": "};
static const lean_object* lp_aesop_List_mapTR_loop___at___00Aesop_RuleTac_tacGenImpl_spec__3___closed__0 = (const lean_object*)&lp_aesop_List_mapTR_loop___at___00Aesop_RuleTac_tacGenImpl_spec__3___closed__0_value;
static lean_once_cell_t lp_aesop_List_mapTR_loop___at___00Aesop_RuleTac_tacGenImpl_spec__3___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_List_mapTR_loop___at___00Aesop_RuleTac_tacGenImpl_spec__3___closed__1;
LEAN_EXPORT lean_object* lp_aesop_List_mapTR_loop___at___00Aesop_RuleTac_tacGenImpl_spec__3(lean_object*, lean_object*);
static const lean_string_object lp_aesop_Aesop_RuleTac_tacGenImpl___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 47, .m_capacity = 47, .m_length = 46, .m_data = "Failed to apply any tactics generated. Errors:"};
static const lean_object* lp_aesop_Aesop_RuleTac_tacGenImpl___closed__0 = (const lean_object*)&lp_aesop_Aesop_RuleTac_tacGenImpl___closed__0_value;
static lean_once_cell_t lp_aesop_Aesop_RuleTac_tacGenImpl___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_RuleTac_tacGenImpl___closed__1;
static const lean_string_object lp_aesop_Aesop_RuleTac_tacGenImpl___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "\n"};
static const lean_object* lp_aesop_Aesop_RuleTac_tacGenImpl___closed__2 = (const lean_object*)&lp_aesop_Aesop_RuleTac_tacGenImpl___closed__2_value;
static const lean_ctor_object lp_aesop_Aesop_RuleTac_tacGenImpl___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop_Aesop_RuleTac_tacGenImpl___closed__2_value)}};
static const lean_object* lp_aesop_Aesop_RuleTac_tacGenImpl___closed__3 = (const lean_object*)&lp_aesop_Aesop_RuleTac_tacGenImpl___closed__3_value;
static lean_once_cell_t lp_aesop_Aesop_RuleTac_tacGenImpl___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_RuleTac_tacGenImpl___closed__4;
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTac_tacGenImpl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTac_tacGenImpl___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_RuleTac_tacticMImpl___lam__0(lean_object* v_x_1_){
_start:
{
uint8_t v___x_2_; 
v___x_2_ = 0;
return v___x_2_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTac_tacticMImpl___lam__0___boxed(lean_object* v_x_3_){
_start:
{
uint8_t v_res_4_; lean_object* v_r_5_; 
v_res_4_ = lp_aesop_Aesop_RuleTac_tacticMImpl___lam__0(v_x_3_);
lean_dec(v_x_3_);
v_r_5_ = lean_box(v_res_4_);
return v_r_5_;
}
}
static lean_object* _init_lp_aesop_Lean_Elab_throwAbortCommand___at___00Lean_evalConst___at___00Aesop_RuleTac_tacticMImpl_spec__0_spec__1___redArg___closed__0(void){
_start:
{
lean_object* v___x_6_; lean_object* v___x_7_; lean_object* v___x_8_; 
v___x_6_ = lean_box(0);
v___x_7_ = l_Lean_Elab_abortCommandExceptionId;
v___x_8_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_8_, 0, v___x_7_);
lean_ctor_set(v___x_8_, 1, v___x_6_);
return v___x_8_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Elab_throwAbortCommand___at___00Lean_evalConst___at___00Aesop_RuleTac_tacticMImpl_spec__0_spec__1___redArg(){
_start:
{
lean_object* v___x_10_; lean_object* v___x_11_; 
v___x_10_ = lean_obj_once(&lp_aesop_Lean_Elab_throwAbortCommand___at___00Lean_evalConst___at___00Aesop_RuleTac_tacticMImpl_spec__0_spec__1___redArg___closed__0, &lp_aesop_Lean_Elab_throwAbortCommand___at___00Lean_evalConst___at___00Aesop_RuleTac_tacticMImpl_spec__0_spec__1___redArg___closed__0_once, _init_lp_aesop_Lean_Elab_throwAbortCommand___at___00Lean_evalConst___at___00Aesop_RuleTac_tacticMImpl_spec__0_spec__1___redArg___closed__0);
v___x_11_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_11_, 0, v___x_10_);
return v___x_11_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Elab_throwAbortCommand___at___00Lean_evalConst___at___00Aesop_RuleTac_tacticMImpl_spec__0_spec__1___redArg___boxed(lean_object* v___y_12_){
_start:
{
lean_object* v_res_13_; 
v_res_13_ = lp_aesop_Lean_Elab_throwAbortCommand___at___00Lean_evalConst___at___00Aesop_RuleTac_tacticMImpl_spec__0_spec__1___redArg();
return v_res_13_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_ofExcept___at___00Lean_evalConst___at___00Aesop_RuleTac_tacticMImpl_spec__0_spec__0_spec__1_spec__4(lean_object* v_msgData_14_, lean_object* v___y_15_, lean_object* v___y_16_, lean_object* v___y_17_, lean_object* v___y_18_){
_start:
{
lean_object* v___x_20_; lean_object* v_env_21_; lean_object* v___x_22_; lean_object* v_mctx_23_; lean_object* v_lctx_24_; lean_object* v_options_25_; lean_object* v___x_26_; lean_object* v___x_27_; lean_object* v___x_28_; 
v___x_20_ = lean_st_ref_get(v___y_18_);
v_env_21_ = lean_ctor_get(v___x_20_, 0);
lean_inc_ref(v_env_21_);
lean_dec(v___x_20_);
v___x_22_ = lean_st_ref_get(v___y_16_);
v_mctx_23_ = lean_ctor_get(v___x_22_, 0);
lean_inc_ref(v_mctx_23_);
lean_dec(v___x_22_);
v_lctx_24_ = lean_ctor_get(v___y_15_, 2);
v_options_25_ = lean_ctor_get(v___y_17_, 2);
lean_inc_ref(v_options_25_);
lean_inc_ref(v_lctx_24_);
v___x_26_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_26_, 0, v_env_21_);
lean_ctor_set(v___x_26_, 1, v_mctx_23_);
lean_ctor_set(v___x_26_, 2, v_lctx_24_);
lean_ctor_set(v___x_26_, 3, v_options_25_);
v___x_27_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_27_, 0, v___x_26_);
lean_ctor_set(v___x_27_, 1, v_msgData_14_);
v___x_28_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_28_, 0, v___x_27_);
return v___x_28_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_ofExcept___at___00Lean_evalConst___at___00Aesop_RuleTac_tacticMImpl_spec__0_spec__0_spec__1_spec__4___boxed(lean_object* v_msgData_29_, lean_object* v___y_30_, lean_object* v___y_31_, lean_object* v___y_32_, lean_object* v___y_33_, lean_object* v___y_34_){
_start:
{
lean_object* v_res_35_; 
v_res_35_ = lp_aesop_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_ofExcept___at___00Lean_evalConst___at___00Aesop_RuleTac_tacticMImpl_spec__0_spec__0_spec__1_spec__4(v_msgData_29_, v___y_30_, v___y_31_, v___y_32_, v___y_33_);
lean_dec(v___y_33_);
lean_dec_ref(v___y_32_);
lean_dec(v___y_31_);
lean_dec_ref(v___y_30_);
return v_res_35_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Lean_ofExcept___at___00Lean_evalConst___at___00Aesop_RuleTac_tacticMImpl_spec__0_spec__0_spec__1___redArg(lean_object* v_msg_36_, lean_object* v___y_37_, lean_object* v___y_38_, lean_object* v___y_39_, lean_object* v___y_40_){
_start:
{
lean_object* v_ref_42_; lean_object* v___x_43_; lean_object* v_a_44_; lean_object* v___x_46_; uint8_t v_isShared_47_; uint8_t v_isSharedCheck_52_; 
v_ref_42_ = lean_ctor_get(v___y_39_, 5);
v___x_43_ = lp_aesop_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_ofExcept___at___00Lean_evalConst___at___00Aesop_RuleTac_tacticMImpl_spec__0_spec__0_spec__1_spec__4(v_msg_36_, v___y_37_, v___y_38_, v___y_39_, v___y_40_);
v_a_44_ = lean_ctor_get(v___x_43_, 0);
v_isSharedCheck_52_ = !lean_is_exclusive(v___x_43_);
if (v_isSharedCheck_52_ == 0)
{
v___x_46_ = v___x_43_;
v_isShared_47_ = v_isSharedCheck_52_;
goto v_resetjp_45_;
}
else
{
lean_inc(v_a_44_);
lean_dec(v___x_43_);
v___x_46_ = lean_box(0);
v_isShared_47_ = v_isSharedCheck_52_;
goto v_resetjp_45_;
}
v_resetjp_45_:
{
lean_object* v___x_48_; lean_object* v___x_50_; 
lean_inc(v_ref_42_);
v___x_48_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_48_, 0, v_ref_42_);
lean_ctor_set(v___x_48_, 1, v_a_44_);
if (v_isShared_47_ == 0)
{
lean_ctor_set_tag(v___x_46_, 1);
lean_ctor_set(v___x_46_, 0, v___x_48_);
v___x_50_ = v___x_46_;
goto v_reusejp_49_;
}
else
{
lean_object* v_reuseFailAlloc_51_; 
v_reuseFailAlloc_51_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_51_, 0, v___x_48_);
v___x_50_ = v_reuseFailAlloc_51_;
goto v_reusejp_49_;
}
v_reusejp_49_:
{
return v___x_50_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Lean_ofExcept___at___00Lean_evalConst___at___00Aesop_RuleTac_tacticMImpl_spec__0_spec__0_spec__1___redArg___boxed(lean_object* v_msg_53_, lean_object* v___y_54_, lean_object* v___y_55_, lean_object* v___y_56_, lean_object* v___y_57_, lean_object* v___y_58_){
_start:
{
lean_object* v_res_59_; 
v_res_59_ = lp_aesop_Lean_throwError___at___00Lean_ofExcept___at___00Lean_evalConst___at___00Aesop_RuleTac_tacticMImpl_spec__0_spec__0_spec__1___redArg(v_msg_53_, v___y_54_, v___y_55_, v___y_56_, v___y_57_);
lean_dec(v___y_57_);
lean_dec_ref(v___y_56_);
lean_dec(v___y_55_);
lean_dec_ref(v___y_54_);
return v_res_59_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_ofExcept___at___00Lean_evalConst___at___00Aesop_RuleTac_tacticMImpl_spec__0_spec__0___redArg(lean_object* v_x_60_, lean_object* v___y_61_, lean_object* v___y_62_, lean_object* v___y_63_, lean_object* v___y_64_, lean_object* v___y_65_){
_start:
{
if (lean_obj_tag(v_x_60_) == 0)
{
lean_object* v_a_67_; lean_object* v___x_68_; lean_object* v___x_69_; 
v_a_67_ = lean_ctor_get(v_x_60_, 0);
lean_inc(v_a_67_);
lean_dec_ref_known(v_x_60_, 1);
v___x_68_ = l_Lean_stringToMessageData(v_a_67_);
v___x_69_ = lp_aesop_Lean_throwError___at___00Lean_ofExcept___at___00Lean_evalConst___at___00Aesop_RuleTac_tacticMImpl_spec__0_spec__0_spec__1___redArg(v___x_68_, v___y_62_, v___y_63_, v___y_64_, v___y_65_);
return v___x_69_;
}
else
{
lean_object* v_a_70_; lean_object* v___x_72_; uint8_t v_isShared_73_; uint8_t v_isSharedCheck_77_; 
v_a_70_ = lean_ctor_get(v_x_60_, 0);
v_isSharedCheck_77_ = !lean_is_exclusive(v_x_60_);
if (v_isSharedCheck_77_ == 0)
{
v___x_72_ = v_x_60_;
v_isShared_73_ = v_isSharedCheck_77_;
goto v_resetjp_71_;
}
else
{
lean_inc(v_a_70_);
lean_dec(v_x_60_);
v___x_72_ = lean_box(0);
v_isShared_73_ = v_isSharedCheck_77_;
goto v_resetjp_71_;
}
v_resetjp_71_:
{
lean_object* v___x_75_; 
if (v_isShared_73_ == 0)
{
lean_ctor_set_tag(v___x_72_, 0);
v___x_75_ = v___x_72_;
goto v_reusejp_74_;
}
else
{
lean_object* v_reuseFailAlloc_76_; 
v_reuseFailAlloc_76_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_76_, 0, v_a_70_);
v___x_75_ = v_reuseFailAlloc_76_;
goto v_reusejp_74_;
}
v_reusejp_74_:
{
return v___x_75_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_ofExcept___at___00Lean_evalConst___at___00Aesop_RuleTac_tacticMImpl_spec__0_spec__0___redArg___boxed(lean_object* v_x_78_, lean_object* v___y_79_, lean_object* v___y_80_, lean_object* v___y_81_, lean_object* v___y_82_, lean_object* v___y_83_, lean_object* v___y_84_){
_start:
{
lean_object* v_res_85_; 
v_res_85_ = lp_aesop_Lean_ofExcept___at___00Lean_evalConst___at___00Aesop_RuleTac_tacticMImpl_spec__0_spec__0___redArg(v_x_78_, v___y_79_, v___y_80_, v___y_81_, v___y_82_, v___y_83_);
lean_dec(v___y_83_);
lean_dec_ref(v___y_82_);
lean_dec(v___y_81_);
lean_dec_ref(v___y_80_);
lean_dec(v___y_79_);
return v_res_85_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_evalConst___at___00Aesop_RuleTac_tacticMImpl_spec__0___redArg(lean_object* v_constName_86_, uint8_t v_checkMeta_87_, lean_object* v___y_88_, lean_object* v___y_89_, lean_object* v___y_90_, lean_object* v___y_91_, lean_object* v___y_92_){
_start:
{
lean_object* v___x_94_; lean_object* v_env_95_; uint8_t v___x_96_; 
v___x_94_ = lean_st_ref_get(v___y_92_);
v_env_95_ = lean_ctor_get(v___x_94_, 0);
lean_inc_ref(v_env_95_);
lean_dec(v___x_94_);
lean_inc(v_constName_86_);
v___x_96_ = lean_has_compile_error(v_env_95_, v_constName_86_);
if (v___x_96_ == 0)
{
lean_object* v___x_97_; lean_object* v_env_98_; lean_object* v_options_99_; lean_object* v___x_100_; lean_object* v___x_101_; 
v___x_97_ = lean_st_ref_get(v___y_92_);
v_env_98_ = lean_ctor_get(v___x_97_, 0);
lean_inc_ref(v_env_98_);
lean_dec(v___x_97_);
v_options_99_ = lean_ctor_get(v___y_91_, 2);
v___x_100_ = l_Lean_Environment_evalConst___redArg(v_env_98_, v_options_99_, v_constName_86_, v_checkMeta_87_);
lean_dec(v_constName_86_);
lean_dec_ref(v_env_98_);
v___x_101_ = lp_aesop_Lean_ofExcept___at___00Lean_evalConst___at___00Aesop_RuleTac_tacticMImpl_spec__0_spec__0___redArg(v___x_100_, v___y_88_, v___y_89_, v___y_90_, v___y_91_, v___y_92_);
return v___x_101_;
}
else
{
lean_object* v___x_102_; 
v___x_102_ = lp_aesop_Lean_Elab_throwAbortCommand___at___00Lean_evalConst___at___00Aesop_RuleTac_tacticMImpl_spec__0_spec__1___redArg();
if (lean_obj_tag(v___x_102_) == 0)
{
lean_object* v___x_103_; lean_object* v_env_104_; lean_object* v_options_105_; lean_object* v___x_106_; lean_object* v___x_107_; 
lean_dec_ref_known(v___x_102_, 1);
v___x_103_ = lean_st_ref_get(v___y_92_);
v_env_104_ = lean_ctor_get(v___x_103_, 0);
lean_inc_ref(v_env_104_);
lean_dec(v___x_103_);
v_options_105_ = lean_ctor_get(v___y_91_, 2);
v___x_106_ = l_Lean_Environment_evalConst___redArg(v_env_104_, v_options_105_, v_constName_86_, v_checkMeta_87_);
lean_dec(v_constName_86_);
lean_dec_ref(v_env_104_);
v___x_107_ = lp_aesop_Lean_ofExcept___at___00Lean_evalConst___at___00Aesop_RuleTac_tacticMImpl_spec__0_spec__0___redArg(v___x_106_, v___y_88_, v___y_89_, v___y_90_, v___y_91_, v___y_92_);
return v___x_107_;
}
else
{
lean_object* v_a_108_; lean_object* v___x_110_; uint8_t v_isShared_111_; uint8_t v_isSharedCheck_115_; 
lean_dec(v_constName_86_);
v_a_108_ = lean_ctor_get(v___x_102_, 0);
v_isSharedCheck_115_ = !lean_is_exclusive(v___x_102_);
if (v_isSharedCheck_115_ == 0)
{
v___x_110_ = v___x_102_;
v_isShared_111_ = v_isSharedCheck_115_;
goto v_resetjp_109_;
}
else
{
lean_inc(v_a_108_);
lean_dec(v___x_102_);
v___x_110_ = lean_box(0);
v_isShared_111_ = v_isSharedCheck_115_;
goto v_resetjp_109_;
}
v_resetjp_109_:
{
lean_object* v___x_113_; 
if (v_isShared_111_ == 0)
{
v___x_113_ = v___x_110_;
goto v_reusejp_112_;
}
else
{
lean_object* v_reuseFailAlloc_114_; 
v_reuseFailAlloc_114_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_114_, 0, v_a_108_);
v___x_113_ = v_reuseFailAlloc_114_;
goto v_reusejp_112_;
}
v_reusejp_112_:
{
return v___x_113_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_evalConst___at___00Aesop_RuleTac_tacticMImpl_spec__0___redArg___boxed(lean_object* v_constName_116_, lean_object* v_checkMeta_117_, lean_object* v___y_118_, lean_object* v___y_119_, lean_object* v___y_120_, lean_object* v___y_121_, lean_object* v___y_122_, lean_object* v___y_123_){
_start:
{
uint8_t v_checkMeta_boxed_124_; lean_object* v_res_125_; 
v_checkMeta_boxed_124_ = lean_unbox(v_checkMeta_117_);
v_res_125_ = lp_aesop_Lean_evalConst___at___00Aesop_RuleTac_tacticMImpl_spec__0___redArg(v_constName_116_, v_checkMeta_boxed_124_, v___y_118_, v___y_119_, v___y_120_, v___y_121_, v___y_122_);
lean_dec(v___y_122_);
lean_dec_ref(v___y_121_);
lean_dec(v___y_120_);
lean_dec_ref(v___y_119_);
lean_dec(v___y_118_);
return v_res_125_;
}
}
LEAN_EXPORT lean_object* lp_aesop_List_mapM_loop___at___00Aesop_RuleTac_tacticMImpl_spec__1(lean_object* v___x_126_, lean_object* v_x_127_, lean_object* v_x_128_, lean_object* v___y_129_, lean_object* v___y_130_, lean_object* v___y_131_, lean_object* v___y_132_, lean_object* v___y_133_){
_start:
{
if (lean_obj_tag(v_x_127_) == 0)
{
lean_object* v___x_135_; lean_object* v___x_136_; 
lean_dec(v___x_126_);
v___x_135_ = l_List_reverse___redArg(v_x_128_);
v___x_136_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_136_, 0, v___x_135_);
return v___x_136_;
}
else
{
lean_object* v_head_137_; lean_object* v_tail_138_; lean_object* v___x_140_; uint8_t v_isShared_141_; uint8_t v_isSharedCheck_156_; 
v_head_137_ = lean_ctor_get(v_x_127_, 0);
v_tail_138_ = lean_ctor_get(v_x_127_, 1);
v_isSharedCheck_156_ = !lean_is_exclusive(v_x_127_);
if (v_isSharedCheck_156_ == 0)
{
v___x_140_ = v_x_127_;
v_isShared_141_ = v_isSharedCheck_156_;
goto v_resetjp_139_;
}
else
{
lean_inc(v_tail_138_);
lean_inc(v_head_137_);
lean_dec(v_x_127_);
v___x_140_ = lean_box(0);
v_isShared_141_ = v_isSharedCheck_156_;
goto v_resetjp_139_;
}
v_resetjp_139_:
{
lean_object* v___x_142_; 
lean_inc(v___x_126_);
v___x_142_ = lp_aesop_Aesop_mvarIdToSubgoal(v___x_126_, v_head_137_, v___y_129_, v___y_130_, v___y_131_, v___y_132_, v___y_133_);
if (lean_obj_tag(v___x_142_) == 0)
{
lean_object* v_a_143_; lean_object* v___x_145_; 
v_a_143_ = lean_ctor_get(v___x_142_, 0);
lean_inc(v_a_143_);
lean_dec_ref_known(v___x_142_, 1);
if (v_isShared_141_ == 0)
{
lean_ctor_set(v___x_140_, 1, v_x_128_);
lean_ctor_set(v___x_140_, 0, v_a_143_);
v___x_145_ = v___x_140_;
goto v_reusejp_144_;
}
else
{
lean_object* v_reuseFailAlloc_147_; 
v_reuseFailAlloc_147_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_147_, 0, v_a_143_);
lean_ctor_set(v_reuseFailAlloc_147_, 1, v_x_128_);
v___x_145_ = v_reuseFailAlloc_147_;
goto v_reusejp_144_;
}
v_reusejp_144_:
{
v_x_127_ = v_tail_138_;
v_x_128_ = v___x_145_;
goto _start;
}
}
else
{
lean_object* v_a_148_; lean_object* v___x_150_; uint8_t v_isShared_151_; uint8_t v_isSharedCheck_155_; 
lean_del_object(v___x_140_);
lean_dec(v_tail_138_);
lean_dec(v_x_128_);
lean_dec(v___x_126_);
v_a_148_ = lean_ctor_get(v___x_142_, 0);
v_isSharedCheck_155_ = !lean_is_exclusive(v___x_142_);
if (v_isSharedCheck_155_ == 0)
{
v___x_150_ = v___x_142_;
v_isShared_151_ = v_isSharedCheck_155_;
goto v_resetjp_149_;
}
else
{
lean_inc(v_a_148_);
lean_dec(v___x_142_);
v___x_150_ = lean_box(0);
v_isShared_151_ = v_isSharedCheck_155_;
goto v_resetjp_149_;
}
v_resetjp_149_:
{
lean_object* v___x_153_; 
if (v_isShared_151_ == 0)
{
v___x_153_ = v___x_150_;
goto v_reusejp_152_;
}
else
{
lean_object* v_reuseFailAlloc_154_; 
v_reuseFailAlloc_154_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_154_, 0, v_a_148_);
v___x_153_ = v_reuseFailAlloc_154_;
goto v_reusejp_152_;
}
v_reusejp_152_:
{
return v___x_153_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_List_mapM_loop___at___00Aesop_RuleTac_tacticMImpl_spec__1___boxed(lean_object* v___x_157_, lean_object* v_x_158_, lean_object* v_x_159_, lean_object* v___y_160_, lean_object* v___y_161_, lean_object* v___y_162_, lean_object* v___y_163_, lean_object* v___y_164_, lean_object* v___y_165_){
_start:
{
lean_object* v_res_166_; 
v_res_166_ = lp_aesop_List_mapM_loop___at___00Aesop_RuleTac_tacticMImpl_spec__1(v___x_157_, v_x_158_, v_x_159_, v___y_160_, v___y_161_, v___y_162_, v___y_163_, v___y_164_);
lean_dec(v___y_164_);
lean_dec_ref(v___y_163_);
lean_dec(v___y_162_);
lean_dec_ref(v___y_161_);
lean_dec(v___y_160_);
return v_res_166_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTac_tacticMImpl(lean_object* v_decl_181_, lean_object* v_a_182_, lean_object* v_a_183_, lean_object* v_a_184_, lean_object* v_a_185_, lean_object* v_a_186_, lean_object* v_a_187_){
_start:
{
uint8_t v___x_189_; lean_object* v___x_190_; 
v___x_189_ = 1;
v___x_190_ = lp_aesop_Lean_evalConst___at___00Aesop_RuleTac_tacticMImpl_spec__0___redArg(v_decl_181_, v___x_189_, v_a_183_, v_a_184_, v_a_185_, v_a_186_, v_a_187_);
if (lean_obj_tag(v___x_190_) == 0)
{
lean_object* v_a_191_; lean_object* v_goal_192_; lean_object* v___x_193_; lean_object* v___x_194_; lean_object* v___x_195_; lean_object* v___x_196_; lean_object* v___x_197_; lean_object* v___x_198_; 
v_a_191_ = lean_ctor_get(v___x_190_, 0);
lean_inc(v_a_191_);
lean_dec_ref_known(v___x_190_, 1);
v_goal_192_ = lean_ctor_get(v_a_182_, 0);
lean_inc_n(v_goal_192_, 2);
lean_dec_ref(v_a_182_);
v___x_193_ = lean_alloc_closure((void*)(l_Lean_Elab_Tactic_run___boxed), 9, 2);
lean_closure_set(v___x_193_, 0, v_goal_192_);
lean_closure_set(v___x_193_, 1, v_a_191_);
v___x_194_ = lean_box(0);
v___x_195_ = lean_box(0);
v___x_196_ = ((lean_object*)(lp_aesop_Aesop_RuleTac_tacticMImpl___closed__2));
v___x_197_ = ((lean_object*)(lp_aesop_Aesop_RuleTac_tacticMImpl___closed__3));
v___x_198_ = l_Lean_Elab_Term_TermElabM_run___redArg(v___x_193_, v___x_196_, v___x_197_, v_a_184_, v_a_185_, v_a_186_, v_a_187_);
if (lean_obj_tag(v___x_198_) == 0)
{
lean_object* v_a_199_; lean_object* v_fst_200_; lean_object* v___x_201_; 
v_a_199_ = lean_ctor_get(v___x_198_, 0);
lean_inc(v_a_199_);
lean_dec_ref_known(v___x_198_, 1);
v_fst_200_ = lean_ctor_get(v_a_199_, 0);
lean_inc(v_fst_200_);
lean_dec(v_a_199_);
v___x_201_ = lp_aesop_List_mapM_loop___at___00Aesop_RuleTac_tacticMImpl_spec__1(v_goal_192_, v_fst_200_, v___x_195_, v_a_183_, v_a_184_, v_a_185_, v_a_186_, v_a_187_);
if (lean_obj_tag(v___x_201_) == 0)
{
lean_object* v_a_202_; lean_object* v___x_203_; 
v_a_202_ = lean_ctor_get(v___x_201_, 0);
lean_inc(v_a_202_);
lean_dec_ref_known(v___x_201_, 1);
v___x_203_ = l_Lean_Meta_saveState___redArg(v_a_185_, v_a_187_);
if (lean_obj_tag(v___x_203_) == 0)
{
lean_object* v_a_204_; lean_object* v___x_206_; uint8_t v_isShared_207_; uint8_t v_isSharedCheck_216_; 
v_a_204_ = lean_ctor_get(v___x_203_, 0);
v_isSharedCheck_216_ = !lean_is_exclusive(v___x_203_);
if (v_isSharedCheck_216_ == 0)
{
v___x_206_ = v___x_203_;
v_isShared_207_ = v_isSharedCheck_216_;
goto v_resetjp_205_;
}
else
{
lean_inc(v_a_204_);
lean_dec(v___x_203_);
v___x_206_ = lean_box(0);
v_isShared_207_ = v_isSharedCheck_216_;
goto v_resetjp_205_;
}
v_resetjp_205_:
{
lean_object* v___x_208_; lean_object* v___x_209_; lean_object* v___x_210_; lean_object* v___x_211_; lean_object* v___x_212_; lean_object* v___x_214_; 
v___x_208_ = lean_array_mk(v_a_202_);
v___x_209_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_209_, 0, v___x_208_);
lean_ctor_set(v___x_209_, 1, v_a_204_);
lean_ctor_set(v___x_209_, 2, v___x_194_);
lean_ctor_set(v___x_209_, 3, v___x_194_);
v___x_210_ = lean_unsigned_to_nat(1u);
v___x_211_ = lean_mk_empty_array_with_capacity(v___x_210_);
v___x_212_ = lean_array_push(v___x_211_, v___x_209_);
if (v_isShared_207_ == 0)
{
lean_ctor_set(v___x_206_, 0, v___x_212_);
v___x_214_ = v___x_206_;
goto v_reusejp_213_;
}
else
{
lean_object* v_reuseFailAlloc_215_; 
v_reuseFailAlloc_215_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_215_, 0, v___x_212_);
v___x_214_ = v_reuseFailAlloc_215_;
goto v_reusejp_213_;
}
v_reusejp_213_:
{
return v___x_214_;
}
}
}
else
{
lean_object* v_a_217_; lean_object* v___x_219_; uint8_t v_isShared_220_; uint8_t v_isSharedCheck_224_; 
lean_dec(v_a_202_);
v_a_217_ = lean_ctor_get(v___x_203_, 0);
v_isSharedCheck_224_ = !lean_is_exclusive(v___x_203_);
if (v_isSharedCheck_224_ == 0)
{
v___x_219_ = v___x_203_;
v_isShared_220_ = v_isSharedCheck_224_;
goto v_resetjp_218_;
}
else
{
lean_inc(v_a_217_);
lean_dec(v___x_203_);
v___x_219_ = lean_box(0);
v_isShared_220_ = v_isSharedCheck_224_;
goto v_resetjp_218_;
}
v_resetjp_218_:
{
lean_object* v___x_222_; 
if (v_isShared_220_ == 0)
{
v___x_222_ = v___x_219_;
goto v_reusejp_221_;
}
else
{
lean_object* v_reuseFailAlloc_223_; 
v_reuseFailAlloc_223_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_223_, 0, v_a_217_);
v___x_222_ = v_reuseFailAlloc_223_;
goto v_reusejp_221_;
}
v_reusejp_221_:
{
return v___x_222_;
}
}
}
}
else
{
lean_object* v_a_225_; lean_object* v___x_227_; uint8_t v_isShared_228_; uint8_t v_isSharedCheck_232_; 
v_a_225_ = lean_ctor_get(v___x_201_, 0);
v_isSharedCheck_232_ = !lean_is_exclusive(v___x_201_);
if (v_isSharedCheck_232_ == 0)
{
v___x_227_ = v___x_201_;
v_isShared_228_ = v_isSharedCheck_232_;
goto v_resetjp_226_;
}
else
{
lean_inc(v_a_225_);
lean_dec(v___x_201_);
v___x_227_ = lean_box(0);
v_isShared_228_ = v_isSharedCheck_232_;
goto v_resetjp_226_;
}
v_resetjp_226_:
{
lean_object* v___x_230_; 
if (v_isShared_228_ == 0)
{
v___x_230_ = v___x_227_;
goto v_reusejp_229_;
}
else
{
lean_object* v_reuseFailAlloc_231_; 
v_reuseFailAlloc_231_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_231_, 0, v_a_225_);
v___x_230_ = v_reuseFailAlloc_231_;
goto v_reusejp_229_;
}
v_reusejp_229_:
{
return v___x_230_;
}
}
}
}
else
{
lean_object* v_a_233_; lean_object* v___x_235_; uint8_t v_isShared_236_; uint8_t v_isSharedCheck_240_; 
lean_dec(v_goal_192_);
v_a_233_ = lean_ctor_get(v___x_198_, 0);
v_isSharedCheck_240_ = !lean_is_exclusive(v___x_198_);
if (v_isSharedCheck_240_ == 0)
{
v___x_235_ = v___x_198_;
v_isShared_236_ = v_isSharedCheck_240_;
goto v_resetjp_234_;
}
else
{
lean_inc(v_a_233_);
lean_dec(v___x_198_);
v___x_235_ = lean_box(0);
v_isShared_236_ = v_isSharedCheck_240_;
goto v_resetjp_234_;
}
v_resetjp_234_:
{
lean_object* v___x_238_; 
if (v_isShared_236_ == 0)
{
v___x_238_ = v___x_235_;
goto v_reusejp_237_;
}
else
{
lean_object* v_reuseFailAlloc_239_; 
v_reuseFailAlloc_239_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_239_, 0, v_a_233_);
v___x_238_ = v_reuseFailAlloc_239_;
goto v_reusejp_237_;
}
v_reusejp_237_:
{
return v___x_238_;
}
}
}
}
else
{
lean_object* v_a_241_; lean_object* v___x_243_; uint8_t v_isShared_244_; uint8_t v_isSharedCheck_248_; 
lean_dec_ref(v_a_182_);
v_a_241_ = lean_ctor_get(v___x_190_, 0);
v_isSharedCheck_248_ = !lean_is_exclusive(v___x_190_);
if (v_isSharedCheck_248_ == 0)
{
v___x_243_ = v___x_190_;
v_isShared_244_ = v_isSharedCheck_248_;
goto v_resetjp_242_;
}
else
{
lean_inc(v_a_241_);
lean_dec(v___x_190_);
v___x_243_ = lean_box(0);
v_isShared_244_ = v_isSharedCheck_248_;
goto v_resetjp_242_;
}
v_resetjp_242_:
{
lean_object* v___x_246_; 
if (v_isShared_244_ == 0)
{
v___x_246_ = v___x_243_;
goto v_reusejp_245_;
}
else
{
lean_object* v_reuseFailAlloc_247_; 
v_reuseFailAlloc_247_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_247_, 0, v_a_241_);
v___x_246_ = v_reuseFailAlloc_247_;
goto v_reusejp_245_;
}
v_reusejp_245_:
{
return v___x_246_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTac_tacticMImpl___boxed(lean_object* v_decl_249_, lean_object* v_a_250_, lean_object* v_a_251_, lean_object* v_a_252_, lean_object* v_a_253_, lean_object* v_a_254_, lean_object* v_a_255_, lean_object* v_a_256_){
_start:
{
lean_object* v_res_257_; 
v_res_257_ = lp_aesop_Aesop_RuleTac_tacticMImpl(v_decl_249_, v_a_250_, v_a_251_, v_a_252_, v_a_253_, v_a_254_, v_a_255_);
lean_dec(v_a_255_);
lean_dec_ref(v_a_254_);
lean_dec(v_a_253_);
lean_dec_ref(v_a_252_);
lean_dec(v_a_251_);
return v_res_257_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Elab_throwAbortCommand___at___00Lean_evalConst___at___00Aesop_RuleTac_tacticMImpl_spec__0_spec__1(lean_object* v_00_u03b1_258_, lean_object* v___y_259_, lean_object* v___y_260_, lean_object* v___y_261_, lean_object* v___y_262_, lean_object* v___y_263_){
_start:
{
lean_object* v___x_265_; 
v___x_265_ = lp_aesop_Lean_Elab_throwAbortCommand___at___00Lean_evalConst___at___00Aesop_RuleTac_tacticMImpl_spec__0_spec__1___redArg();
return v___x_265_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Elab_throwAbortCommand___at___00Lean_evalConst___at___00Aesop_RuleTac_tacticMImpl_spec__0_spec__1___boxed(lean_object* v_00_u03b1_266_, lean_object* v___y_267_, lean_object* v___y_268_, lean_object* v___y_269_, lean_object* v___y_270_, lean_object* v___y_271_, lean_object* v___y_272_){
_start:
{
lean_object* v_res_273_; 
v_res_273_ = lp_aesop_Lean_Elab_throwAbortCommand___at___00Lean_evalConst___at___00Aesop_RuleTac_tacticMImpl_spec__0_spec__1(v_00_u03b1_266_, v___y_267_, v___y_268_, v___y_269_, v___y_270_, v___y_271_);
lean_dec(v___y_271_);
lean_dec_ref(v___y_270_);
lean_dec(v___y_269_);
lean_dec_ref(v___y_268_);
lean_dec(v___y_267_);
return v_res_273_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_evalConst___at___00Aesop_RuleTac_tacticMImpl_spec__0(lean_object* v_00_u03b1_274_, lean_object* v_constName_275_, uint8_t v_checkMeta_276_, lean_object* v___y_277_, lean_object* v___y_278_, lean_object* v___y_279_, lean_object* v___y_280_, lean_object* v___y_281_){
_start:
{
lean_object* v___x_283_; 
v___x_283_ = lp_aesop_Lean_evalConst___at___00Aesop_RuleTac_tacticMImpl_spec__0___redArg(v_constName_275_, v_checkMeta_276_, v___y_277_, v___y_278_, v___y_279_, v___y_280_, v___y_281_);
return v___x_283_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_evalConst___at___00Aesop_RuleTac_tacticMImpl_spec__0___boxed(lean_object* v_00_u03b1_284_, lean_object* v_constName_285_, lean_object* v_checkMeta_286_, lean_object* v___y_287_, lean_object* v___y_288_, lean_object* v___y_289_, lean_object* v___y_290_, lean_object* v___y_291_, lean_object* v___y_292_){
_start:
{
uint8_t v_checkMeta_boxed_293_; lean_object* v_res_294_; 
v_checkMeta_boxed_293_ = lean_unbox(v_checkMeta_286_);
v_res_294_ = lp_aesop_Lean_evalConst___at___00Aesop_RuleTac_tacticMImpl_spec__0(v_00_u03b1_284_, v_constName_285_, v_checkMeta_boxed_293_, v___y_287_, v___y_288_, v___y_289_, v___y_290_, v___y_291_);
lean_dec(v___y_291_);
lean_dec_ref(v___y_290_);
lean_dec(v___y_289_);
lean_dec_ref(v___y_288_);
lean_dec(v___y_287_);
return v_res_294_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_ofExcept___at___00Lean_evalConst___at___00Aesop_RuleTac_tacticMImpl_spec__0_spec__0(lean_object* v_00_u03b1_295_, lean_object* v_x_296_, lean_object* v___y_297_, lean_object* v___y_298_, lean_object* v___y_299_, lean_object* v___y_300_, lean_object* v___y_301_){
_start:
{
lean_object* v___x_303_; 
v___x_303_ = lp_aesop_Lean_ofExcept___at___00Lean_evalConst___at___00Aesop_RuleTac_tacticMImpl_spec__0_spec__0___redArg(v_x_296_, v___y_297_, v___y_298_, v___y_299_, v___y_300_, v___y_301_);
return v___x_303_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_ofExcept___at___00Lean_evalConst___at___00Aesop_RuleTac_tacticMImpl_spec__0_spec__0___boxed(lean_object* v_00_u03b1_304_, lean_object* v_x_305_, lean_object* v___y_306_, lean_object* v___y_307_, lean_object* v___y_308_, lean_object* v___y_309_, lean_object* v___y_310_, lean_object* v___y_311_){
_start:
{
lean_object* v_res_312_; 
v_res_312_ = lp_aesop_Lean_ofExcept___at___00Lean_evalConst___at___00Aesop_RuleTac_tacticMImpl_spec__0_spec__0(v_00_u03b1_304_, v_x_305_, v___y_306_, v___y_307_, v___y_308_, v___y_309_, v___y_310_);
lean_dec(v___y_310_);
lean_dec_ref(v___y_309_);
lean_dec(v___y_308_);
lean_dec_ref(v___y_307_);
lean_dec(v___y_306_);
return v_res_312_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Lean_ofExcept___at___00Lean_evalConst___at___00Aesop_RuleTac_tacticMImpl_spec__0_spec__0_spec__1(lean_object* v_00_u03b1_313_, lean_object* v_msg_314_, lean_object* v___y_315_, lean_object* v___y_316_, lean_object* v___y_317_, lean_object* v___y_318_, lean_object* v___y_319_){
_start:
{
lean_object* v___x_321_; 
v___x_321_ = lp_aesop_Lean_throwError___at___00Lean_ofExcept___at___00Lean_evalConst___at___00Aesop_RuleTac_tacticMImpl_spec__0_spec__0_spec__1___redArg(v_msg_314_, v___y_316_, v___y_317_, v___y_318_, v___y_319_);
return v___x_321_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Lean_ofExcept___at___00Lean_evalConst___at___00Aesop_RuleTac_tacticMImpl_spec__0_spec__0_spec__1___boxed(lean_object* v_00_u03b1_322_, lean_object* v_msg_323_, lean_object* v___y_324_, lean_object* v___y_325_, lean_object* v___y_326_, lean_object* v___y_327_, lean_object* v___y_328_, lean_object* v___y_329_){
_start:
{
lean_object* v_res_330_; 
v_res_330_ = lp_aesop_Lean_throwError___at___00Lean_ofExcept___at___00Lean_evalConst___at___00Aesop_RuleTac_tacticMImpl_spec__0_spec__0_spec__1(v_00_u03b1_322_, v_msg_323_, v___y_324_, v___y_325_, v___y_326_, v___y_327_, v___y_328_);
lean_dec(v___y_328_);
lean_dec_ref(v___y_327_);
lean_dec(v___y_326_);
lean_dec_ref(v___y_325_);
lean_dec(v___y_324_);
return v_res_330_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTac_ruleTacImpl(lean_object* v_decl_331_, lean_object* v_input_332_, lean_object* v_a_333_, lean_object* v_a_334_, lean_object* v_a_335_, lean_object* v_a_336_, lean_object* v_a_337_){
_start:
{
uint8_t v___x_339_; lean_object* v___x_340_; 
v___x_339_ = 1;
v___x_340_ = lp_aesop_Lean_evalConst___at___00Aesop_RuleTac_tacticMImpl_spec__0___redArg(v_decl_331_, v___x_339_, v_a_333_, v_a_334_, v_a_335_, v_a_336_, v_a_337_);
if (lean_obj_tag(v___x_340_) == 0)
{
lean_object* v_a_341_; lean_object* v___x_342_; 
v_a_341_ = lean_ctor_get(v___x_340_, 0);
lean_inc(v_a_341_);
lean_dec_ref_known(v___x_340_, 1);
lean_inc(v_a_337_);
lean_inc_ref(v_a_336_);
lean_inc(v_a_335_);
lean_inc_ref(v_a_334_);
lean_inc(v_a_333_);
v___x_342_ = lean_apply_7(v_a_341_, v_input_332_, v_a_333_, v_a_334_, v_a_335_, v_a_336_, v_a_337_, lean_box(0));
return v___x_342_;
}
else
{
lean_object* v_a_343_; lean_object* v___x_345_; uint8_t v_isShared_346_; uint8_t v_isSharedCheck_350_; 
lean_dec_ref(v_input_332_);
v_a_343_ = lean_ctor_get(v___x_340_, 0);
v_isSharedCheck_350_ = !lean_is_exclusive(v___x_340_);
if (v_isSharedCheck_350_ == 0)
{
v___x_345_ = v___x_340_;
v_isShared_346_ = v_isSharedCheck_350_;
goto v_resetjp_344_;
}
else
{
lean_inc(v_a_343_);
lean_dec(v___x_340_);
v___x_345_ = lean_box(0);
v_isShared_346_ = v_isSharedCheck_350_;
goto v_resetjp_344_;
}
v_resetjp_344_:
{
lean_object* v___x_348_; 
if (v_isShared_346_ == 0)
{
v___x_348_ = v___x_345_;
goto v_reusejp_347_;
}
else
{
lean_object* v_reuseFailAlloc_349_; 
v_reuseFailAlloc_349_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_349_, 0, v_a_343_);
v___x_348_ = v_reuseFailAlloc_349_;
goto v_reusejp_347_;
}
v_reusejp_347_:
{
return v___x_348_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTac_ruleTacImpl___boxed(lean_object* v_decl_351_, lean_object* v_input_352_, lean_object* v_a_353_, lean_object* v_a_354_, lean_object* v_a_355_, lean_object* v_a_356_, lean_object* v_a_357_, lean_object* v_a_358_){
_start:
{
lean_object* v_res_359_; 
v_res_359_ = lp_aesop_Aesop_RuleTac_ruleTacImpl(v_decl_351_, v_input_352_, v_a_353_, v_a_354_, v_a_355_, v_a_356_, v_a_357_);
lean_dec(v_a_357_);
lean_dec_ref(v_a_356_);
lean_dec(v_a_355_);
lean_dec_ref(v_a_354_);
lean_dec(v_a_353_);
return v_res_359_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTac_singleRuleTacImpl(lean_object* v_decl_360_, lean_object* v_a_361_, lean_object* v_a_362_, lean_object* v_a_363_, lean_object* v_a_364_, lean_object* v_a_365_, lean_object* v_a_366_){
_start:
{
uint8_t v___x_368_; lean_object* v___x_369_; 
v___x_368_ = 1;
v___x_369_ = lp_aesop_Lean_evalConst___at___00Aesop_RuleTac_tacticMImpl_spec__0___redArg(v_decl_360_, v___x_368_, v_a_362_, v_a_363_, v_a_364_, v_a_365_, v_a_366_);
if (lean_obj_tag(v___x_369_) == 0)
{
lean_object* v_a_370_; lean_object* v___x_371_; 
v_a_370_ = lean_ctor_get(v___x_369_, 0);
lean_inc(v_a_370_);
lean_dec_ref_known(v___x_369_, 1);
lean_inc(v_a_366_);
lean_inc_ref(v_a_365_);
lean_inc(v_a_364_);
lean_inc_ref(v_a_363_);
lean_inc(v_a_362_);
v___x_371_ = lean_apply_7(v_a_370_, v_a_361_, v_a_362_, v_a_363_, v_a_364_, v_a_365_, v_a_366_, lean_box(0));
if (lean_obj_tag(v___x_371_) == 0)
{
lean_object* v_a_372_; lean_object* v_snd_373_; lean_object* v_fst_374_; lean_object* v_fst_375_; lean_object* v_snd_376_; lean_object* v___x_377_; 
v_a_372_ = lean_ctor_get(v___x_371_, 0);
lean_inc(v_a_372_);
lean_dec_ref_known(v___x_371_, 1);
v_snd_373_ = lean_ctor_get(v_a_372_, 1);
lean_inc(v_snd_373_);
v_fst_374_ = lean_ctor_get(v_a_372_, 0);
lean_inc(v_fst_374_);
lean_dec(v_a_372_);
v_fst_375_ = lean_ctor_get(v_snd_373_, 0);
lean_inc(v_fst_375_);
v_snd_376_ = lean_ctor_get(v_snd_373_, 1);
lean_inc(v_snd_376_);
lean_dec(v_snd_373_);
v___x_377_ = l_Lean_Meta_saveState___redArg(v_a_364_, v_a_366_);
if (lean_obj_tag(v___x_377_) == 0)
{
lean_object* v_a_378_; lean_object* v___x_380_; uint8_t v_isShared_381_; uint8_t v_isSharedCheck_389_; 
v_a_378_ = lean_ctor_get(v___x_377_, 0);
v_isSharedCheck_389_ = !lean_is_exclusive(v___x_377_);
if (v_isSharedCheck_389_ == 0)
{
v___x_380_ = v___x_377_;
v_isShared_381_ = v_isSharedCheck_389_;
goto v_resetjp_379_;
}
else
{
lean_inc(v_a_378_);
lean_dec(v___x_377_);
v___x_380_ = lean_box(0);
v_isShared_381_ = v_isSharedCheck_389_;
goto v_resetjp_379_;
}
v_resetjp_379_:
{
lean_object* v___x_382_; lean_object* v___x_383_; lean_object* v___x_384_; lean_object* v___x_385_; lean_object* v___x_387_; 
v___x_382_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_382_, 0, v_fst_374_);
lean_ctor_set(v___x_382_, 1, v_a_378_);
lean_ctor_set(v___x_382_, 2, v_fst_375_);
lean_ctor_set(v___x_382_, 3, v_snd_376_);
v___x_383_ = lean_unsigned_to_nat(1u);
v___x_384_ = lean_mk_empty_array_with_capacity(v___x_383_);
v___x_385_ = lean_array_push(v___x_384_, v___x_382_);
if (v_isShared_381_ == 0)
{
lean_ctor_set(v___x_380_, 0, v___x_385_);
v___x_387_ = v___x_380_;
goto v_reusejp_386_;
}
else
{
lean_object* v_reuseFailAlloc_388_; 
v_reuseFailAlloc_388_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_388_, 0, v___x_385_);
v___x_387_ = v_reuseFailAlloc_388_;
goto v_reusejp_386_;
}
v_reusejp_386_:
{
return v___x_387_;
}
}
}
else
{
lean_object* v_a_390_; lean_object* v___x_392_; uint8_t v_isShared_393_; uint8_t v_isSharedCheck_397_; 
lean_dec(v_snd_376_);
lean_dec(v_fst_375_);
lean_dec(v_fst_374_);
v_a_390_ = lean_ctor_get(v___x_377_, 0);
v_isSharedCheck_397_ = !lean_is_exclusive(v___x_377_);
if (v_isSharedCheck_397_ == 0)
{
v___x_392_ = v___x_377_;
v_isShared_393_ = v_isSharedCheck_397_;
goto v_resetjp_391_;
}
else
{
lean_inc(v_a_390_);
lean_dec(v___x_377_);
v___x_392_ = lean_box(0);
v_isShared_393_ = v_isSharedCheck_397_;
goto v_resetjp_391_;
}
v_resetjp_391_:
{
lean_object* v___x_395_; 
if (v_isShared_393_ == 0)
{
v___x_395_ = v___x_392_;
goto v_reusejp_394_;
}
else
{
lean_object* v_reuseFailAlloc_396_; 
v_reuseFailAlloc_396_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_396_, 0, v_a_390_);
v___x_395_ = v_reuseFailAlloc_396_;
goto v_reusejp_394_;
}
v_reusejp_394_:
{
return v___x_395_;
}
}
}
}
else
{
lean_object* v_a_398_; lean_object* v___x_400_; uint8_t v_isShared_401_; uint8_t v_isSharedCheck_405_; 
v_a_398_ = lean_ctor_get(v___x_371_, 0);
v_isSharedCheck_405_ = !lean_is_exclusive(v___x_371_);
if (v_isSharedCheck_405_ == 0)
{
v___x_400_ = v___x_371_;
v_isShared_401_ = v_isSharedCheck_405_;
goto v_resetjp_399_;
}
else
{
lean_inc(v_a_398_);
lean_dec(v___x_371_);
v___x_400_ = lean_box(0);
v_isShared_401_ = v_isSharedCheck_405_;
goto v_resetjp_399_;
}
v_resetjp_399_:
{
lean_object* v___x_403_; 
if (v_isShared_401_ == 0)
{
v___x_403_ = v___x_400_;
goto v_reusejp_402_;
}
else
{
lean_object* v_reuseFailAlloc_404_; 
v_reuseFailAlloc_404_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_404_, 0, v_a_398_);
v___x_403_ = v_reuseFailAlloc_404_;
goto v_reusejp_402_;
}
v_reusejp_402_:
{
return v___x_403_;
}
}
}
}
else
{
lean_object* v_a_406_; lean_object* v___x_408_; uint8_t v_isShared_409_; uint8_t v_isSharedCheck_413_; 
lean_dec_ref(v_a_361_);
v_a_406_ = lean_ctor_get(v___x_369_, 0);
v_isSharedCheck_413_ = !lean_is_exclusive(v___x_369_);
if (v_isSharedCheck_413_ == 0)
{
v___x_408_ = v___x_369_;
v_isShared_409_ = v_isSharedCheck_413_;
goto v_resetjp_407_;
}
else
{
lean_inc(v_a_406_);
lean_dec(v___x_369_);
v___x_408_ = lean_box(0);
v_isShared_409_ = v_isSharedCheck_413_;
goto v_resetjp_407_;
}
v_resetjp_407_:
{
lean_object* v___x_411_; 
if (v_isShared_409_ == 0)
{
v___x_411_ = v___x_408_;
goto v_reusejp_410_;
}
else
{
lean_object* v_reuseFailAlloc_412_; 
v_reuseFailAlloc_412_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_412_, 0, v_a_406_);
v___x_411_ = v_reuseFailAlloc_412_;
goto v_reusejp_410_;
}
v_reusejp_410_:
{
return v___x_411_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTac_singleRuleTacImpl___boxed(lean_object* v_decl_414_, lean_object* v_a_415_, lean_object* v_a_416_, lean_object* v_a_417_, lean_object* v_a_418_, lean_object* v_a_419_, lean_object* v_a_420_, lean_object* v_a_421_){
_start:
{
lean_object* v_res_422_; 
v_res_422_ = lp_aesop_Aesop_RuleTac_singleRuleTacImpl(v_decl_414_, v_a_415_, v_a_416_, v_a_417_, v_a_418_, v_a_419_, v_a_420_);
lean_dec(v_a_420_);
lean_dec_ref(v_a_419_);
lean_dec(v_a_418_);
lean_dec_ref(v_a_417_);
lean_dec(v_a_416_);
return v_res_422_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_RuleTac_tacticStx___lam__0(lean_object* v_x_423_){
_start:
{
uint8_t v___x_424_; 
v___x_424_ = 0;
return v___x_424_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTac_tacticStx___lam__0___boxed(lean_object* v_x_425_){
_start:
{
uint8_t v_res_426_; lean_object* v_r_427_; 
v_res_426_ = lp_aesop_Aesop_RuleTac_tacticStx___lam__0(v_x_425_);
lean_dec(v_x_425_);
v_r_427_ = lean_box(v_res_426_);
return v_r_427_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_RuleTac_tacticStx_spec__0___redArg(lean_object* v_msg_428_, lean_object* v___y_429_, lean_object* v___y_430_, lean_object* v___y_431_, lean_object* v___y_432_){
_start:
{
lean_object* v_ref_434_; lean_object* v___x_435_; lean_object* v_a_436_; lean_object* v___x_438_; uint8_t v_isShared_439_; uint8_t v_isSharedCheck_444_; 
v_ref_434_ = lean_ctor_get(v___y_431_, 5);
v___x_435_ = lp_aesop_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_ofExcept___at___00Lean_evalConst___at___00Aesop_RuleTac_tacticMImpl_spec__0_spec__0_spec__1_spec__4(v_msg_428_, v___y_429_, v___y_430_, v___y_431_, v___y_432_);
v_a_436_ = lean_ctor_get(v___x_435_, 0);
v_isSharedCheck_444_ = !lean_is_exclusive(v___x_435_);
if (v_isSharedCheck_444_ == 0)
{
v___x_438_ = v___x_435_;
v_isShared_439_ = v_isSharedCheck_444_;
goto v_resetjp_437_;
}
else
{
lean_inc(v_a_436_);
lean_dec(v___x_435_);
v___x_438_ = lean_box(0);
v_isShared_439_ = v_isSharedCheck_444_;
goto v_resetjp_437_;
}
v_resetjp_437_:
{
lean_object* v___x_440_; lean_object* v___x_442_; 
lean_inc(v_ref_434_);
v___x_440_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_440_, 0, v_ref_434_);
lean_ctor_set(v___x_440_, 1, v_a_436_);
if (v_isShared_439_ == 0)
{
lean_ctor_set_tag(v___x_438_, 1);
lean_ctor_set(v___x_438_, 0, v___x_440_);
v___x_442_ = v___x_438_;
goto v_reusejp_441_;
}
else
{
lean_object* v_reuseFailAlloc_443_; 
v_reuseFailAlloc_443_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_443_, 0, v___x_440_);
v___x_442_ = v_reuseFailAlloc_443_;
goto v_reusejp_441_;
}
v_reusejp_441_:
{
return v___x_442_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_RuleTac_tacticStx_spec__0___redArg___boxed(lean_object* v_msg_445_, lean_object* v___y_446_, lean_object* v___y_447_, lean_object* v___y_448_, lean_object* v___y_449_, lean_object* v___y_450_){
_start:
{
lean_object* v_res_451_; 
v_res_451_ = lp_aesop_Lean_throwError___at___00Aesop_RuleTac_tacticStx_spec__0___redArg(v_msg_445_, v___y_446_, v___y_447_, v___y_448_, v___y_449_);
lean_dec(v___y_449_);
lean_dec_ref(v___y_448_);
lean_dec(v___y_447_);
lean_dec_ref(v___y_446_);
return v_res_451_;
}
}
static lean_object* _init_lp_aesop_Aesop_RuleTac_tacticStx___lam__1___closed__6(void){
_start:
{
lean_object* v___x_462_; lean_object* v___x_463_; 
v___x_462_ = ((lean_object*)(lp_aesop_Aesop_RuleTac_tacticStx___lam__1___closed__5));
v___x_463_ = l_Lean_stringToMessageData(v___x_462_);
return v___x_463_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTac_tacticStx___lam__1(uint8_t v___x_472_, lean_object* v_stx_473_, uint8_t v___x_474_, lean_object* v___y_475_, lean_object* v___y_476_, lean_object* v___y_477_, lean_object* v___y_478_){
_start:
{
if (v___x_472_ == 0)
{
lean_object* v___x_480_; uint8_t v___x_481_; 
v___x_480_ = ((lean_object*)(lp_aesop_Aesop_RuleTac_tacticStx___lam__1___closed__4));
lean_inc(v_stx_473_);
v___x_481_ = l_Lean_Syntax_isOfKind(v_stx_473_, v___x_480_);
if (v___x_481_ == 0)
{
lean_object* v___x_482_; lean_object* v___x_483_; 
lean_dec(v_stx_473_);
v___x_482_ = lean_obj_once(&lp_aesop_Aesop_RuleTac_tacticStx___lam__1___closed__6, &lp_aesop_Aesop_RuleTac_tacticStx___lam__1___closed__6_once, _init_lp_aesop_Aesop_RuleTac_tacticStx___lam__1___closed__6);
v___x_483_ = lp_aesop_Lean_throwError___at___00Aesop_RuleTac_tacticStx_spec__0___redArg(v___x_482_, v___y_475_, v___y_476_, v___y_477_, v___y_478_);
return v___x_483_;
}
else
{
lean_object* v_ref_484_; lean_object* v___x_485_; lean_object* v___x_486_; lean_object* v___x_487_; lean_object* v___x_488_; lean_object* v___x_489_; lean_object* v___x_490_; lean_object* v___x_491_; lean_object* v___x_492_; lean_object* v___x_493_; 
v_ref_484_ = lean_ctor_get(v___y_477_, 5);
v___x_485_ = l_Lean_SourceInfo_fromRef(v_ref_484_, v___x_474_);
v___x_486_ = ((lean_object*)(lp_aesop_Aesop_RuleTac_tacticStx___lam__1___closed__8));
v___x_487_ = ((lean_object*)(lp_aesop_Aesop_RuleTac_tacticStx___lam__1___closed__9));
lean_inc_n(v___x_485_, 2);
v___x_488_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_488_, 0, v___x_485_);
lean_ctor_set(v___x_488_, 1, v___x_487_);
v___x_489_ = ((lean_object*)(lp_aesop_Aesop_RuleTac_tacticStx___lam__1___closed__10));
v___x_490_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_490_, 0, v___x_485_);
lean_ctor_set(v___x_490_, 1, v___x_489_);
v___x_491_ = l_Lean_Syntax_node3(v___x_485_, v___x_486_, v___x_488_, v_stx_473_, v___x_490_);
v___x_492_ = lp_aesop_Aesop_Script_Tactic_unstructured(v___x_491_);
v___x_493_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_493_, 0, v___x_492_);
return v___x_493_;
}
}
else
{
lean_object* v___x_494_; lean_object* v___x_495_; 
v___x_494_ = lp_aesop_Aesop_Script_Tactic_unstructured(v_stx_473_);
v___x_495_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_495_, 0, v___x_494_);
return v___x_495_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTac_tacticStx___lam__1___boxed(lean_object* v___x_496_, lean_object* v_stx_497_, lean_object* v___x_498_, lean_object* v___y_499_, lean_object* v___y_500_, lean_object* v___y_501_, lean_object* v___y_502_, lean_object* v___y_503_){
_start:
{
uint8_t v___x_5259__boxed_504_; uint8_t v___x_5260__boxed_505_; lean_object* v_res_506_; 
v___x_5259__boxed_504_ = lean_unbox(v___x_496_);
v___x_5260__boxed_505_ = lean_unbox(v___x_498_);
v_res_506_ = lp_aesop_Aesop_RuleTac_tacticStx___lam__1(v___x_5259__boxed_504_, v_stx_497_, v___x_5260__boxed_505_, v___y_499_, v___y_500_, v___y_501_, v___y_502_);
lean_dec(v___y_502_);
lean_dec_ref(v___y_501_);
lean_dec(v___y_500_);
lean_dec_ref(v___y_499_);
return v_res_506_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_RuleTac_tacticStx_spec__1(lean_object* v___x_507_, size_t v_sz_508_, size_t v_i_509_, lean_object* v_bs_510_, lean_object* v___y_511_, lean_object* v___y_512_, lean_object* v___y_513_, lean_object* v___y_514_, lean_object* v___y_515_){
_start:
{
uint8_t v___x_517_; 
v___x_517_ = lean_usize_dec_lt(v_i_509_, v_sz_508_);
if (v___x_517_ == 0)
{
lean_object* v___x_518_; 
lean_dec(v___x_507_);
v___x_518_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_518_, 0, v_bs_510_);
return v___x_518_;
}
else
{
lean_object* v_v_519_; lean_object* v___x_520_; 
v_v_519_ = lean_array_uget_borrowed(v_bs_510_, v_i_509_);
lean_inc(v_v_519_);
lean_inc(v___x_507_);
v___x_520_ = lp_aesop_Aesop_mvarIdToSubgoal(v___x_507_, v_v_519_, v___y_511_, v___y_512_, v___y_513_, v___y_514_, v___y_515_);
if (lean_obj_tag(v___x_520_) == 0)
{
lean_object* v_a_521_; lean_object* v___x_522_; lean_object* v_bs_x27_523_; size_t v___x_524_; size_t v___x_525_; lean_object* v___x_526_; 
v_a_521_ = lean_ctor_get(v___x_520_, 0);
lean_inc(v_a_521_);
lean_dec_ref_known(v___x_520_, 1);
v___x_522_ = lean_unsigned_to_nat(0u);
v_bs_x27_523_ = lean_array_uset(v_bs_510_, v_i_509_, v___x_522_);
v___x_524_ = ((size_t)1ULL);
v___x_525_ = lean_usize_add(v_i_509_, v___x_524_);
v___x_526_ = lean_array_uset(v_bs_x27_523_, v_i_509_, v_a_521_);
v_i_509_ = v___x_525_;
v_bs_510_ = v___x_526_;
goto _start;
}
else
{
lean_object* v_a_528_; lean_object* v___x_530_; uint8_t v_isShared_531_; uint8_t v_isSharedCheck_535_; 
lean_dec_ref(v_bs_510_);
lean_dec(v___x_507_);
v_a_528_ = lean_ctor_get(v___x_520_, 0);
v_isSharedCheck_535_ = !lean_is_exclusive(v___x_520_);
if (v_isSharedCheck_535_ == 0)
{
v___x_530_ = v___x_520_;
v_isShared_531_ = v_isSharedCheck_535_;
goto v_resetjp_529_;
}
else
{
lean_inc(v_a_528_);
lean_dec(v___x_520_);
v___x_530_ = lean_box(0);
v_isShared_531_ = v_isSharedCheck_535_;
goto v_resetjp_529_;
}
v_resetjp_529_:
{
lean_object* v___x_533_; 
if (v_isShared_531_ == 0)
{
v___x_533_ = v___x_530_;
goto v_reusejp_532_;
}
else
{
lean_object* v_reuseFailAlloc_534_; 
v_reuseFailAlloc_534_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_534_, 0, v_a_528_);
v___x_533_ = v_reuseFailAlloc_534_;
goto v_reusejp_532_;
}
v_reusejp_532_:
{
return v___x_533_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_RuleTac_tacticStx_spec__1___boxed(lean_object* v___x_536_, lean_object* v_sz_537_, lean_object* v_i_538_, lean_object* v_bs_539_, lean_object* v___y_540_, lean_object* v___y_541_, lean_object* v___y_542_, lean_object* v___y_543_, lean_object* v___y_544_, lean_object* v___y_545_){
_start:
{
size_t v_sz_boxed_546_; size_t v_i_boxed_547_; lean_object* v_res_548_; 
v_sz_boxed_546_ = lean_unbox_usize(v_sz_537_);
lean_dec(v_sz_537_);
v_i_boxed_547_ = lean_unbox_usize(v_i_538_);
lean_dec(v_i_538_);
v_res_548_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_RuleTac_tacticStx_spec__1(v___x_536_, v_sz_boxed_546_, v_i_boxed_547_, v_bs_539_, v___y_540_, v___y_541_, v___y_542_, v___y_543_, v___y_544_);
lean_dec(v___y_544_);
lean_dec_ref(v___y_543_);
lean_dec(v___y_542_);
lean_dec_ref(v___y_541_);
lean_dec(v___y_540_);
return v_res_548_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTac_tacticStx(lean_object* v_stx_561_, lean_object* v_a_562_, lean_object* v_a_563_, lean_object* v_a_564_, lean_object* v_a_565_, lean_object* v_a_566_, lean_object* v_a_567_){
_start:
{
lean_object* v___x_569_; 
v___x_569_ = l_Lean_Meta_saveState___redArg(v_a_565_, v_a_567_);
if (lean_obj_tag(v___x_569_) == 0)
{
lean_object* v_a_570_; lean_object* v_goal_571_; lean_object* v___x_573_; uint8_t v_isShared_574_; uint8_t v_isSharedCheck_648_; 
v_a_570_ = lean_ctor_get(v___x_569_, 0);
lean_inc(v_a_570_);
lean_dec_ref_known(v___x_569_, 1);
v_goal_571_ = lean_ctor_get(v_a_562_, 0);
v_isSharedCheck_648_ = !lean_is_exclusive(v_a_562_);
if (v_isSharedCheck_648_ == 0)
{
lean_object* v_unused_649_; lean_object* v_unused_650_; lean_object* v_unused_651_; lean_object* v_unused_652_; 
v_unused_649_ = lean_ctor_get(v_a_562_, 4);
lean_dec(v_unused_649_);
v_unused_650_ = lean_ctor_get(v_a_562_, 3);
lean_dec(v_unused_650_);
v_unused_651_ = lean_ctor_get(v_a_562_, 2);
lean_dec(v_unused_651_);
v_unused_652_ = lean_ctor_get(v_a_562_, 1);
lean_dec(v_unused_652_);
v___x_573_ = v_a_562_;
v_isShared_574_ = v_isSharedCheck_648_;
goto v_resetjp_572_;
}
else
{
lean_inc(v_goal_571_);
lean_dec(v_a_562_);
v___x_573_ = lean_box(0);
v_isShared_574_ = v_isSharedCheck_648_;
goto v_resetjp_572_;
}
v_resetjp_572_:
{
lean_object* v___x_575_; lean_object* v___x_576_; lean_object* v___x_577_; lean_object* v___x_578_; uint8_t v___x_579_; lean_object* v___x_580_; lean_object* v___x_581_; lean_object* v___x_582_; 
lean_inc(v_stx_561_);
v___x_575_ = lean_alloc_closure((void*)(l_Lean_Elab_Tactic_evalTactic___boxed), 10, 1);
lean_closure_set(v___x_575_, 0, v_stx_561_);
v___x_576_ = lean_alloc_closure((void*)(l_Lean_Elab_Tactic_withoutRecover___boxed), 11, 2);
lean_closure_set(v___x_576_, 0, lean_box(0));
lean_closure_set(v___x_576_, 1, v___x_575_);
lean_inc(v_goal_571_);
v___x_577_ = lean_alloc_closure((void*)(l_Lean_Elab_Tactic_run___boxed), 9, 2);
lean_closure_set(v___x_577_, 0, v_goal_571_);
lean_closure_set(v___x_577_, 1, v___x_576_);
v___x_578_ = lean_box(0);
v___x_579_ = 0;
v___x_580_ = ((lean_object*)(lp_aesop_Aesop_RuleTac_tacticStx___closed__1));
v___x_581_ = ((lean_object*)(lp_aesop_Aesop_RuleTac_tacticMImpl___closed__3));
v___x_582_ = l_Lean_Elab_Term_TermElabM_run___redArg(v___x_577_, v___x_580_, v___x_581_, v_a_564_, v_a_565_, v_a_566_, v_a_567_);
if (lean_obj_tag(v___x_582_) == 0)
{
lean_object* v_a_583_; lean_object* v___x_584_; 
v_a_583_ = lean_ctor_get(v___x_582_, 0);
lean_inc(v_a_583_);
lean_dec_ref_known(v___x_582_, 1);
v___x_584_ = l_Lean_Meta_saveState___redArg(v_a_565_, v_a_567_);
if (lean_obj_tag(v___x_584_) == 0)
{
lean_object* v_a_585_; lean_object* v_fst_586_; lean_object* v___x_587_; lean_object* v___x_588_; uint8_t v___x_589_; lean_object* v___x_590_; lean_object* v___x_591_; lean_object* v___y_592_; lean_object* v___x_593_; lean_object* v___x_594_; lean_object* v___x_595_; lean_object* v___x_597_; 
v_a_585_ = lean_ctor_get(v___x_584_, 0);
lean_inc(v_a_585_);
lean_dec_ref_known(v___x_584_, 1);
v_fst_586_ = lean_ctor_get(v_a_583_, 0);
lean_inc(v_fst_586_);
lean_dec(v_a_583_);
v___x_587_ = lean_array_mk(v_fst_586_);
v___x_588_ = ((lean_object*)(lp_aesop_Aesop_RuleTac_tacticStx___closed__3));
lean_inc(v_stx_561_);
v___x_589_ = l_Lean_Syntax_isOfKind(v_stx_561_, v___x_588_);
v___x_590_ = lean_box(v___x_589_);
v___x_591_ = lean_box(v___x_579_);
v___y_592_ = lean_alloc_closure((void*)(lp_aesop_Aesop_RuleTac_tacticStx___lam__1___boxed), 8, 3);
lean_closure_set(v___y_592_, 0, v___x_590_);
lean_closure_set(v___y_592_, 1, v_stx_561_);
lean_closure_set(v___y_592_, 2, v___x_591_);
v___x_593_ = lean_unsigned_to_nat(1u);
v___x_594_ = lean_mk_empty_array_with_capacity(v___x_593_);
lean_inc_ref(v___x_594_);
v___x_595_ = lean_array_push(v___x_594_, v___y_592_);
lean_inc_ref(v___x_587_);
lean_inc(v_goal_571_);
if (v_isShared_574_ == 0)
{
lean_ctor_set(v___x_573_, 4, v___x_587_);
lean_ctor_set(v___x_573_, 3, v_a_585_);
lean_ctor_set(v___x_573_, 2, v___x_595_);
lean_ctor_set(v___x_573_, 1, v_goal_571_);
lean_ctor_set(v___x_573_, 0, v_a_570_);
v___x_597_ = v___x_573_;
goto v_reusejp_596_;
}
else
{
lean_object* v_reuseFailAlloc_631_; 
v_reuseFailAlloc_631_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_631_, 0, v_a_570_);
lean_ctor_set(v_reuseFailAlloc_631_, 1, v_goal_571_);
lean_ctor_set(v_reuseFailAlloc_631_, 2, v___x_595_);
lean_ctor_set(v_reuseFailAlloc_631_, 3, v_a_585_);
lean_ctor_set(v_reuseFailAlloc_631_, 4, v___x_587_);
v___x_597_ = v_reuseFailAlloc_631_;
goto v_reusejp_596_;
}
v_reusejp_596_:
{
size_t v_sz_598_; size_t v___x_599_; lean_object* v___x_600_; 
v_sz_598_ = lean_array_size(v___x_587_);
v___x_599_ = ((size_t)0ULL);
v___x_600_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_RuleTac_tacticStx_spec__1(v_goal_571_, v_sz_598_, v___x_599_, v___x_587_, v_a_563_, v_a_564_, v_a_565_, v_a_566_, v_a_567_);
if (lean_obj_tag(v___x_600_) == 0)
{
lean_object* v_a_601_; lean_object* v___x_602_; 
v_a_601_ = lean_ctor_get(v___x_600_, 0);
lean_inc(v_a_601_);
lean_dec_ref_known(v___x_600_, 1);
v___x_602_ = l_Lean_Meta_saveState___redArg(v_a_565_, v_a_567_);
if (lean_obj_tag(v___x_602_) == 0)
{
lean_object* v_a_603_; lean_object* v___x_605_; uint8_t v_isShared_606_; uint8_t v_isSharedCheck_614_; 
v_a_603_ = lean_ctor_get(v___x_602_, 0);
v_isSharedCheck_614_ = !lean_is_exclusive(v___x_602_);
if (v_isSharedCheck_614_ == 0)
{
v___x_605_ = v___x_602_;
v_isShared_606_ = v_isSharedCheck_614_;
goto v_resetjp_604_;
}
else
{
lean_inc(v_a_603_);
lean_dec(v___x_602_);
v___x_605_ = lean_box(0);
v_isShared_606_ = v_isSharedCheck_614_;
goto v_resetjp_604_;
}
v_resetjp_604_:
{
lean_object* v___x_607_; lean_object* v___x_608_; lean_object* v___x_609_; lean_object* v___x_610_; lean_object* v___x_612_; 
lean_inc_ref(v___x_594_);
v___x_607_ = lean_array_push(v___x_594_, v___x_597_);
v___x_608_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_608_, 0, v___x_607_);
v___x_609_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_609_, 0, v_a_601_);
lean_ctor_set(v___x_609_, 1, v_a_603_);
lean_ctor_set(v___x_609_, 2, v___x_608_);
lean_ctor_set(v___x_609_, 3, v___x_578_);
v___x_610_ = lean_array_push(v___x_594_, v___x_609_);
if (v_isShared_606_ == 0)
{
lean_ctor_set(v___x_605_, 0, v___x_610_);
v___x_612_ = v___x_605_;
goto v_reusejp_611_;
}
else
{
lean_object* v_reuseFailAlloc_613_; 
v_reuseFailAlloc_613_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_613_, 0, v___x_610_);
v___x_612_ = v_reuseFailAlloc_613_;
goto v_reusejp_611_;
}
v_reusejp_611_:
{
return v___x_612_;
}
}
}
else
{
lean_object* v_a_615_; lean_object* v___x_617_; uint8_t v_isShared_618_; uint8_t v_isSharedCheck_622_; 
lean_dec(v_a_601_);
lean_dec_ref(v___x_597_);
lean_dec_ref(v___x_594_);
v_a_615_ = lean_ctor_get(v___x_602_, 0);
v_isSharedCheck_622_ = !lean_is_exclusive(v___x_602_);
if (v_isSharedCheck_622_ == 0)
{
v___x_617_ = v___x_602_;
v_isShared_618_ = v_isSharedCheck_622_;
goto v_resetjp_616_;
}
else
{
lean_inc(v_a_615_);
lean_dec(v___x_602_);
v___x_617_ = lean_box(0);
v_isShared_618_ = v_isSharedCheck_622_;
goto v_resetjp_616_;
}
v_resetjp_616_:
{
lean_object* v___x_620_; 
if (v_isShared_618_ == 0)
{
v___x_620_ = v___x_617_;
goto v_reusejp_619_;
}
else
{
lean_object* v_reuseFailAlloc_621_; 
v_reuseFailAlloc_621_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_621_, 0, v_a_615_);
v___x_620_ = v_reuseFailAlloc_621_;
goto v_reusejp_619_;
}
v_reusejp_619_:
{
return v___x_620_;
}
}
}
}
else
{
lean_object* v_a_623_; lean_object* v___x_625_; uint8_t v_isShared_626_; uint8_t v_isSharedCheck_630_; 
lean_dec_ref(v___x_597_);
lean_dec_ref(v___x_594_);
v_a_623_ = lean_ctor_get(v___x_600_, 0);
v_isSharedCheck_630_ = !lean_is_exclusive(v___x_600_);
if (v_isSharedCheck_630_ == 0)
{
v___x_625_ = v___x_600_;
v_isShared_626_ = v_isSharedCheck_630_;
goto v_resetjp_624_;
}
else
{
lean_inc(v_a_623_);
lean_dec(v___x_600_);
v___x_625_ = lean_box(0);
v_isShared_626_ = v_isSharedCheck_630_;
goto v_resetjp_624_;
}
v_resetjp_624_:
{
lean_object* v___x_628_; 
if (v_isShared_626_ == 0)
{
v___x_628_ = v___x_625_;
goto v_reusejp_627_;
}
else
{
lean_object* v_reuseFailAlloc_629_; 
v_reuseFailAlloc_629_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_629_, 0, v_a_623_);
v___x_628_ = v_reuseFailAlloc_629_;
goto v_reusejp_627_;
}
v_reusejp_627_:
{
return v___x_628_;
}
}
}
}
}
else
{
lean_object* v_a_632_; lean_object* v___x_634_; uint8_t v_isShared_635_; uint8_t v_isSharedCheck_639_; 
lean_dec(v_a_583_);
lean_del_object(v___x_573_);
lean_dec(v_goal_571_);
lean_dec(v_a_570_);
lean_dec(v_stx_561_);
v_a_632_ = lean_ctor_get(v___x_584_, 0);
v_isSharedCheck_639_ = !lean_is_exclusive(v___x_584_);
if (v_isSharedCheck_639_ == 0)
{
v___x_634_ = v___x_584_;
v_isShared_635_ = v_isSharedCheck_639_;
goto v_resetjp_633_;
}
else
{
lean_inc(v_a_632_);
lean_dec(v___x_584_);
v___x_634_ = lean_box(0);
v_isShared_635_ = v_isSharedCheck_639_;
goto v_resetjp_633_;
}
v_resetjp_633_:
{
lean_object* v___x_637_; 
if (v_isShared_635_ == 0)
{
v___x_637_ = v___x_634_;
goto v_reusejp_636_;
}
else
{
lean_object* v_reuseFailAlloc_638_; 
v_reuseFailAlloc_638_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_638_, 0, v_a_632_);
v___x_637_ = v_reuseFailAlloc_638_;
goto v_reusejp_636_;
}
v_reusejp_636_:
{
return v___x_637_;
}
}
}
}
else
{
lean_object* v_a_640_; lean_object* v___x_642_; uint8_t v_isShared_643_; uint8_t v_isSharedCheck_647_; 
lean_del_object(v___x_573_);
lean_dec(v_goal_571_);
lean_dec(v_a_570_);
lean_dec(v_stx_561_);
v_a_640_ = lean_ctor_get(v___x_582_, 0);
v_isSharedCheck_647_ = !lean_is_exclusive(v___x_582_);
if (v_isSharedCheck_647_ == 0)
{
v___x_642_ = v___x_582_;
v_isShared_643_ = v_isSharedCheck_647_;
goto v_resetjp_641_;
}
else
{
lean_inc(v_a_640_);
lean_dec(v___x_582_);
v___x_642_ = lean_box(0);
v_isShared_643_ = v_isSharedCheck_647_;
goto v_resetjp_641_;
}
v_resetjp_641_:
{
lean_object* v___x_645_; 
if (v_isShared_643_ == 0)
{
v___x_645_ = v___x_642_;
goto v_reusejp_644_;
}
else
{
lean_object* v_reuseFailAlloc_646_; 
v_reuseFailAlloc_646_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_646_, 0, v_a_640_);
v___x_645_ = v_reuseFailAlloc_646_;
goto v_reusejp_644_;
}
v_reusejp_644_:
{
return v___x_645_;
}
}
}
}
}
else
{
lean_object* v_a_653_; lean_object* v___x_655_; uint8_t v_isShared_656_; uint8_t v_isSharedCheck_660_; 
lean_dec_ref(v_a_562_);
lean_dec(v_stx_561_);
v_a_653_ = lean_ctor_get(v___x_569_, 0);
v_isSharedCheck_660_ = !lean_is_exclusive(v___x_569_);
if (v_isSharedCheck_660_ == 0)
{
v___x_655_ = v___x_569_;
v_isShared_656_ = v_isSharedCheck_660_;
goto v_resetjp_654_;
}
else
{
lean_inc(v_a_653_);
lean_dec(v___x_569_);
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
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTac_tacticStx___boxed(lean_object* v_stx_661_, lean_object* v_a_662_, lean_object* v_a_663_, lean_object* v_a_664_, lean_object* v_a_665_, lean_object* v_a_666_, lean_object* v_a_667_, lean_object* v_a_668_){
_start:
{
lean_object* v_res_669_; 
v_res_669_ = lp_aesop_Aesop_RuleTac_tacticStx(v_stx_661_, v_a_662_, v_a_663_, v_a_664_, v_a_665_, v_a_666_, v_a_667_);
lean_dec(v_a_667_);
lean_dec_ref(v_a_666_);
lean_dec(v_a_665_);
lean_dec_ref(v_a_664_);
lean_dec(v_a_663_);
return v_res_669_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_RuleTac_tacticStx_spec__0(lean_object* v_00_u03b1_670_, lean_object* v_msg_671_, lean_object* v___y_672_, lean_object* v___y_673_, lean_object* v___y_674_, lean_object* v___y_675_){
_start:
{
lean_object* v___x_677_; 
v___x_677_ = lp_aesop_Lean_throwError___at___00Aesop_RuleTac_tacticStx_spec__0___redArg(v_msg_671_, v___y_672_, v___y_673_, v___y_674_, v___y_675_);
return v___x_677_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_RuleTac_tacticStx_spec__0___boxed(lean_object* v_00_u03b1_678_, lean_object* v_msg_679_, lean_object* v___y_680_, lean_object* v___y_681_, lean_object* v___y_682_, lean_object* v___y_683_, lean_object* v___y_684_){
_start:
{
lean_object* v_res_685_; 
v_res_685_ = lp_aesop_Lean_throwError___at___00Aesop_RuleTac_tacticStx_spec__0(v_00_u03b1_678_, v_msg_679_, v___y_680_, v___y_681_, v___y_682_, v___y_683_);
lean_dec(v___y_683_);
lean_dec_ref(v___y_682_);
lean_dec(v___y_681_);
lean_dec_ref(v___y_680_);
return v_res_685_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_getExprMVarAssignment_x3f___at___00Aesop_RuleTac_tacGenImpl_spec__0___redArg(lean_object* v_mvarId_686_, lean_object* v___y_687_){
_start:
{
lean_object* v___x_689_; lean_object* v_mctx_690_; lean_object* v___x_691_; lean_object* v___x_692_; 
v___x_689_ = lean_st_ref_get(v___y_687_);
v_mctx_690_ = lean_ctor_get(v___x_689_, 0);
lean_inc_ref(v_mctx_690_);
lean_dec(v___x_689_);
v___x_691_ = l_Lean_MetavarContext_getExprAssignmentCore_x3f(v_mctx_690_, v_mvarId_686_);
lean_dec_ref(v_mctx_690_);
v___x_692_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_692_, 0, v___x_691_);
return v___x_692_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_getExprMVarAssignment_x3f___at___00Aesop_RuleTac_tacGenImpl_spec__0___redArg___boxed(lean_object* v_mvarId_693_, lean_object* v___y_694_, lean_object* v___y_695_){
_start:
{
lean_object* v_res_696_; 
v_res_696_ = lp_aesop_Lean_getExprMVarAssignment_x3f___at___00Aesop_RuleTac_tacGenImpl_spec__0___redArg(v_mvarId_693_, v___y_694_);
lean_dec(v___y_694_);
lean_dec(v_mvarId_693_);
return v_res_696_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_getExprMVarAssignment_x3f___at___00Aesop_RuleTac_tacGenImpl_spec__0(lean_object* v_mvarId_697_, lean_object* v___y_698_, lean_object* v___y_699_, lean_object* v___y_700_, lean_object* v___y_701_, lean_object* v___y_702_){
_start:
{
lean_object* v___x_704_; 
v___x_704_ = lp_aesop_Lean_getExprMVarAssignment_x3f___at___00Aesop_RuleTac_tacGenImpl_spec__0___redArg(v_mvarId_697_, v___y_700_);
return v___x_704_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_getExprMVarAssignment_x3f___at___00Aesop_RuleTac_tacGenImpl_spec__0___boxed(lean_object* v_mvarId_705_, lean_object* v___y_706_, lean_object* v___y_707_, lean_object* v___y_708_, lean_object* v___y_709_, lean_object* v___y_710_, lean_object* v___y_711_){
_start:
{
lean_object* v_res_712_; 
v_res_712_ = lp_aesop_Lean_getExprMVarAssignment_x3f___at___00Aesop_RuleTac_tacGenImpl_spec__0(v_mvarId_705_, v___y_706_, v___y_707_, v___y_708_, v___y_709_, v___y_710_);
lean_dec(v___y_710_);
lean_dec_ref(v___y_709_);
lean_dec(v___y_708_);
lean_dec_ref(v___y_707_);
lean_dec(v___y_706_);
lean_dec(v_mvarId_705_);
return v_res_712_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_hasSorry___at___00Aesop_RuleTac_tacGenImpl_spec__1___redArg(lean_object* v_e_713_, lean_object* v___y_714_){
_start:
{
lean_object* v___x_716_; lean_object* v_mctx_717_; uint8_t v___x_718_; lean_object* v___x_719_; lean_object* v___x_720_; 
v___x_716_ = lean_st_ref_get(v___y_714_);
v_mctx_717_ = lean_ctor_get(v___x_716_, 0);
lean_inc_ref(v_mctx_717_);
lean_dec(v___x_716_);
v___x_718_ = lp_aesop___private_Aesop_Util_Basic_0__Aesop_hasSorry_go(v_mctx_717_, v_e_713_);
v___x_719_ = lean_box(v___x_718_);
v___x_720_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_720_, 0, v___x_719_);
return v___x_720_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_hasSorry___at___00Aesop_RuleTac_tacGenImpl_spec__1___redArg___boxed(lean_object* v_e_721_, lean_object* v___y_722_, lean_object* v___y_723_){
_start:
{
lean_object* v_res_724_; 
v_res_724_ = lp_aesop_Aesop_hasSorry___at___00Aesop_RuleTac_tacGenImpl_spec__1___redArg(v_e_721_, v___y_722_);
lean_dec(v___y_722_);
lean_dec_ref(v_e_721_);
return v_res_724_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_hasSorry___at___00Aesop_RuleTac_tacGenImpl_spec__1(lean_object* v_e_725_, lean_object* v___y_726_, lean_object* v___y_727_, lean_object* v___y_728_, lean_object* v___y_729_, lean_object* v___y_730_){
_start:
{
lean_object* v___x_732_; 
v___x_732_ = lp_aesop_Aesop_hasSorry___at___00Aesop_RuleTac_tacGenImpl_spec__1___redArg(v_e_725_, v___y_728_);
return v___x_732_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_hasSorry___at___00Aesop_RuleTac_tacGenImpl_spec__1___boxed(lean_object* v_e_733_, lean_object* v___y_734_, lean_object* v___y_735_, lean_object* v___y_736_, lean_object* v___y_737_, lean_object* v___y_738_, lean_object* v___y_739_){
_start:
{
lean_object* v_res_740_; 
v_res_740_ = lp_aesop_Aesop_hasSorry___at___00Aesop_RuleTac_tacGenImpl_spec__1(v_e_733_, v___y_734_, v___y_735_, v___y_736_, v___y_737_, v___y_738_);
lean_dec(v___y_738_);
lean_dec_ref(v___y_737_);
lean_dec(v___y_736_);
lean_dec_ref(v___y_735_);
lean_dec(v___y_734_);
lean_dec_ref(v_e_733_);
return v_res_740_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RuleTac_tacGenImpl_spec__2___lam__1(lean_object* v___x_741_, lean_object* v___y_742_, lean_object* v___y_743_, lean_object* v___y_744_, lean_object* v___y_745_){
_start:
{
lean_object* v___x_747_; 
v___x_747_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_747_, 0, v___x_741_);
return v___x_747_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RuleTac_tacGenImpl_spec__2___lam__1___boxed(lean_object* v___x_748_, lean_object* v___y_749_, lean_object* v___y_750_, lean_object* v___y_751_, lean_object* v___y_752_, lean_object* v___y_753_){
_start:
{
lean_object* v_res_754_; 
v_res_754_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RuleTac_tacGenImpl_spec__2___lam__1(v___x_748_, v___y_749_, v___y_750_, v___y_751_, v___y_752_);
lean_dec(v___y_752_);
lean_dec_ref(v___y_751_);
lean_dec(v___y_750_);
lean_dec_ref(v___y_749_);
return v_res_754_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RuleTac_tacGenImpl_spec__2___lam__0(lean_object* v_a_755_, lean_object* v_a_756_, lean_object* v___x_757_, lean_object* v_a_758_, lean_object* v___x_759_, lean_object* v___x_760_, lean_object* v_fst_761_, lean_object* v_snd_762_, lean_object* v_____r_763_, lean_object* v___y_764_, lean_object* v___y_765_, lean_object* v___y_766_, lean_object* v___y_767_, lean_object* v___y_768_){
_start:
{
lean_object* v___x_770_; lean_object* v___f_771_; lean_object* v___x_772_; lean_object* v___x_773_; lean_object* v___x_774_; lean_object* v___x_775_; size_t v_sz_776_; size_t v___x_777_; lean_object* v___x_778_; 
v___x_770_ = lp_aesop_Aesop_Script_Tactic_unstructured(v_a_755_);
v___f_771_ = lean_alloc_closure((void*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RuleTac_tacGenImpl_spec__2___lam__1___boxed), 6, 1);
lean_closure_set(v___f_771_, 0, v___x_770_);
v___x_772_ = lean_unsigned_to_nat(1u);
v___x_773_ = lean_mk_empty_array_with_capacity(v___x_772_);
lean_inc_ref(v___x_773_);
v___x_774_ = lean_array_push(v___x_773_, v___f_771_);
lean_inc_ref(v___x_759_);
lean_inc_ref(v_a_758_);
lean_inc(v___x_757_);
v___x_775_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_775_, 0, v_a_756_);
lean_ctor_set(v___x_775_, 1, v___x_757_);
lean_ctor_set(v___x_775_, 2, v___x_774_);
lean_ctor_set(v___x_775_, 3, v_a_758_);
lean_ctor_set(v___x_775_, 4, v___x_759_);
v_sz_776_ = lean_array_size(v___x_759_);
v___x_777_ = ((size_t)0ULL);
v___x_778_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_RuleTac_tacticStx_spec__1(v___x_757_, v_sz_776_, v___x_777_, v___x_759_, v___y_764_, v___y_765_, v___y_766_, v___y_767_, v___y_768_);
if (lean_obj_tag(v___x_778_) == 0)
{
lean_object* v_a_779_; lean_object* v___x_781_; uint8_t v_isShared_782_; uint8_t v_isSharedCheck_793_; 
v_a_779_ = lean_ctor_get(v___x_778_, 0);
v_isSharedCheck_793_ = !lean_is_exclusive(v___x_778_);
if (v_isSharedCheck_793_ == 0)
{
v___x_781_ = v___x_778_;
v_isShared_782_ = v_isSharedCheck_793_;
goto v_resetjp_780_;
}
else
{
lean_inc(v_a_779_);
lean_dec(v___x_778_);
v___x_781_ = lean_box(0);
v_isShared_782_ = v_isSharedCheck_793_;
goto v_resetjp_780_;
}
v_resetjp_780_:
{
lean_object* v___x_783_; lean_object* v___x_784_; lean_object* v___x_785_; lean_object* v___x_786_; lean_object* v___x_787_; lean_object* v___x_788_; lean_object* v___x_789_; lean_object* v___x_791_; 
v___x_783_ = lean_array_push(v___x_773_, v___x_775_);
v___x_784_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_784_, 0, v___x_783_);
v___x_785_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_785_, 0, v_a_779_);
lean_ctor_set(v___x_785_, 1, v_a_758_);
lean_ctor_set(v___x_785_, 2, v___x_784_);
lean_ctor_set(v___x_785_, 3, v___x_760_);
v___x_786_ = lean_array_push(v_fst_761_, v___x_785_);
v___x_787_ = lean_box(0);
v___x_788_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_788_, 0, v___x_786_);
lean_ctor_set(v___x_788_, 1, v_snd_762_);
v___x_789_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_789_, 0, v___x_787_);
lean_ctor_set(v___x_789_, 1, v___x_788_);
if (v_isShared_782_ == 0)
{
lean_ctor_set(v___x_781_, 0, v___x_789_);
v___x_791_ = v___x_781_;
goto v_reusejp_790_;
}
else
{
lean_object* v_reuseFailAlloc_792_; 
v_reuseFailAlloc_792_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_792_, 0, v___x_789_);
v___x_791_ = v_reuseFailAlloc_792_;
goto v_reusejp_790_;
}
v_reusejp_790_:
{
return v___x_791_;
}
}
}
else
{
lean_object* v_a_794_; lean_object* v___x_796_; uint8_t v_isShared_797_; uint8_t v_isSharedCheck_801_; 
lean_dec_ref_known(v___x_775_, 5);
lean_dec_ref(v___x_773_);
lean_dec(v_snd_762_);
lean_dec(v_fst_761_);
lean_dec(v___x_760_);
lean_dec_ref(v_a_758_);
v_a_794_ = lean_ctor_get(v___x_778_, 0);
v_isSharedCheck_801_ = !lean_is_exclusive(v___x_778_);
if (v_isSharedCheck_801_ == 0)
{
v___x_796_ = v___x_778_;
v_isShared_797_ = v_isSharedCheck_801_;
goto v_resetjp_795_;
}
else
{
lean_inc(v_a_794_);
lean_dec(v___x_778_);
v___x_796_ = lean_box(0);
v_isShared_797_ = v_isSharedCheck_801_;
goto v_resetjp_795_;
}
v_resetjp_795_:
{
lean_object* v___x_799_; 
if (v_isShared_797_ == 0)
{
v___x_799_ = v___x_796_;
goto v_reusejp_798_;
}
else
{
lean_object* v_reuseFailAlloc_800_; 
v_reuseFailAlloc_800_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_800_, 0, v_a_794_);
v___x_799_ = v_reuseFailAlloc_800_;
goto v_reusejp_798_;
}
v_reusejp_798_:
{
return v___x_799_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RuleTac_tacGenImpl_spec__2___lam__0___boxed(lean_object* v_a_802_, lean_object* v_a_803_, lean_object* v___x_804_, lean_object* v_a_805_, lean_object* v___x_806_, lean_object* v___x_807_, lean_object* v_fst_808_, lean_object* v_snd_809_, lean_object* v_____r_810_, lean_object* v___y_811_, lean_object* v___y_812_, lean_object* v___y_813_, lean_object* v___y_814_, lean_object* v___y_815_, lean_object* v___y_816_){
_start:
{
lean_object* v_res_817_; 
v_res_817_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RuleTac_tacGenImpl_spec__2___lam__0(v_a_802_, v_a_803_, v___x_804_, v_a_805_, v___x_806_, v___x_807_, v_fst_808_, v_snd_809_, v_____r_810_, v___y_811_, v___y_812_, v___y_813_, v___y_814_, v___y_815_);
lean_dec(v___y_815_);
lean_dec_ref(v___y_814_);
lean_dec(v___y_813_);
lean_dec_ref(v___y_812_);
lean_dec(v___y_811_);
return v_res_817_;
}
}
static lean_object* _init_lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RuleTac_tacGenImpl_spec__2___closed__2(void){
_start:
{
lean_object* v___x_820_; lean_object* v___x_821_; 
v___x_820_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RuleTac_tacGenImpl_spec__2___closed__1));
v___x_821_ = l_Lean_stringToMessageData(v___x_820_);
return v___x_821_;
}
}
static lean_object* _init_lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RuleTac_tacGenImpl_spec__2___closed__4(void){
_start:
{
lean_object* v___x_823_; lean_object* v___x_824_; 
v___x_823_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RuleTac_tacGenImpl_spec__2___closed__3));
v___x_824_ = l_Lean_stringToMessageData(v___x_823_);
return v___x_824_;
}
}
static lean_object* _init_lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RuleTac_tacGenImpl_spec__2___closed__6(void){
_start:
{
lean_object* v___x_826_; lean_object* v___x_827_; 
v___x_826_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RuleTac_tacGenImpl_spec__2___closed__5));
v___x_827_ = l_Lean_stringToMessageData(v___x_826_);
return v___x_827_;
}
}
static lean_object* _init_lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RuleTac_tacGenImpl_spec__2___closed__8(void){
_start:
{
lean_object* v___x_829_; lean_object* v___x_830_; 
v___x_829_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RuleTac_tacGenImpl_spec__2___closed__7));
v___x_830_ = l_Lean_stringToMessageData(v___x_829_);
return v___x_830_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RuleTac_tacGenImpl_spec__2(lean_object* v_a_831_, lean_object* v___x_832_, lean_object* v_as_833_, size_t v_sz_834_, size_t v_i_835_, lean_object* v_b_836_, lean_object* v___y_837_, lean_object* v___y_838_, lean_object* v___y_839_, lean_object* v___y_840_, lean_object* v___y_841_){
_start:
{
lean_object* v_fst_844_; lean_object* v_snd_845_; uint8_t v___x_850_; 
v___x_850_ = lean_usize_dec_lt(v_i_835_, v_sz_834_);
if (v___x_850_ == 0)
{
lean_object* v___x_851_; 
lean_dec(v___x_832_);
lean_dec_ref(v_a_831_);
v___x_851_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_851_, 0, v_b_836_);
return v___x_851_;
}
else
{
lean_object* v_a_852_; lean_object* v_fst_853_; lean_object* v_snd_854_; lean_object* v___x_856_; uint8_t v_isShared_857_; uint8_t v_isSharedCheck_970_; 
v_a_852_ = lean_array_uget(v_as_833_, v_i_835_);
v_fst_853_ = lean_ctor_get(v_a_852_, 0);
v_snd_854_ = lean_ctor_get(v_a_852_, 1);
v_isSharedCheck_970_ = !lean_is_exclusive(v_a_852_);
if (v_isSharedCheck_970_ == 0)
{
v___x_856_ = v_a_852_;
v_isShared_857_ = v_isSharedCheck_970_;
goto v_resetjp_855_;
}
else
{
lean_inc(v_snd_854_);
lean_inc(v_fst_853_);
lean_dec(v_a_852_);
v___x_856_ = lean_box(0);
v_isShared_857_ = v_isSharedCheck_970_;
goto v_resetjp_855_;
}
v_resetjp_855_:
{
lean_object* v___x_858_; 
v___x_858_ = l_Lean_Meta_SavedState_restore___redArg(v_a_831_, v___y_839_, v___y_841_);
if (lean_obj_tag(v___x_858_) == 0)
{
lean_object* v___x_860_; uint8_t v_isShared_861_; uint8_t v_isSharedCheck_960_; 
v_isSharedCheck_960_ = !lean_is_exclusive(v___x_858_);
if (v_isSharedCheck_960_ == 0)
{
lean_object* v_unused_961_; 
v_unused_961_ = lean_ctor_get(v___x_858_, 0);
lean_dec(v_unused_961_);
v___x_860_ = v___x_858_;
v_isShared_861_ = v_isSharedCheck_960_;
goto v_resetjp_859_;
}
else
{
lean_dec(v___x_858_);
v___x_860_ = lean_box(0);
v_isShared_861_ = v_isSharedCheck_960_;
goto v_resetjp_859_;
}
v_resetjp_859_:
{
lean_object* v___x_862_; lean_object* v_fst_863_; lean_object* v_snd_864_; lean_object* v___x_866_; uint8_t v_isShared_867_; uint8_t v_isSharedCheck_959_; 
v___x_862_ = lean_st_ref_get(v___y_841_);
v_fst_863_ = lean_ctor_get(v_b_836_, 0);
v_snd_864_ = lean_ctor_get(v_b_836_, 1);
v_isSharedCheck_959_ = !lean_is_exclusive(v_b_836_);
if (v_isSharedCheck_959_ == 0)
{
v___x_866_ = v_b_836_;
v_isShared_867_ = v_isSharedCheck_959_;
goto v_resetjp_865_;
}
else
{
lean_inc(v_snd_864_);
lean_inc(v_fst_863_);
lean_dec(v_b_836_);
v___x_866_ = lean_box(0);
v_isShared_867_ = v_isSharedCheck_959_;
goto v_resetjp_865_;
}
v_resetjp_865_:
{
lean_object* v___y_869_; uint8_t v___y_870_; lean_object* v_a_879_; lean_object* v___y_883_; double v___x_889_; lean_object* v___x_890_; 
v___x_889_ = lean_unbox_float(v_snd_854_);
v___x_890_ = lp_aesop_Aesop_Percent_ofFloat(v___x_889_);
if (lean_obj_tag(v___x_890_) == 1)
{
lean_object* v_env_891_; lean_object* v___x_892_; lean_object* v___x_893_; lean_object* v___x_894_; 
lean_dec(v_snd_854_);
v_env_891_ = lean_ctor_get(v___x_862_, 0);
lean_inc_ref(v_env_891_);
lean_dec(v___x_862_);
v___x_892_ = ((lean_object*)(lp_aesop_Aesop_RuleTac_tacticStx___closed__3));
v___x_893_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RuleTac_tacGenImpl_spec__2___closed__0));
lean_inc(v_fst_853_);
v___x_894_ = l_Lean_Parser_runParserCategory(v_env_891_, v___x_892_, v_fst_853_, v___x_893_);
if (lean_obj_tag(v___x_894_) == 1)
{
lean_object* v_a_895_; lean_object* v___f_896_; lean_object* v___x_897_; lean_object* v___x_898_; lean_object* v___x_899_; lean_object* v___x_900_; lean_object* v___x_901_; uint8_t v___x_902_; lean_object* v___x_903_; lean_object* v___x_904_; lean_object* v___x_905_; lean_object* v___x_906_; 
lean_del_object(v___x_856_);
v_a_895_ = lean_ctor_get(v___x_894_, 0);
lean_inc_n(v_a_895_, 2);
lean_dec_ref_known(v___x_894_, 1);
v___f_896_ = ((lean_object*)(lp_aesop_Aesop_RuleTac_tacticMImpl___closed__0));
v___x_897_ = lean_alloc_closure((void*)(l_Lean_Elab_Tactic_evalTactic___boxed), 10, 1);
lean_closure_set(v___x_897_, 0, v_a_895_);
lean_inc(v___x_832_);
v___x_898_ = lean_alloc_closure((void*)(l_Lean_Elab_Tactic_run___boxed), 9, 2);
lean_closure_set(v___x_898_, 0, v___x_832_);
lean_closure_set(v___x_898_, 1, v___x_897_);
v___x_899_ = lean_box(0);
v___x_900_ = lean_box(0);
v___x_901_ = lean_box(1);
v___x_902_ = 0;
v___x_903_ = ((lean_object*)(lp_aesop_Aesop_RuleTac_tacticMImpl___closed__1));
v___x_904_ = lean_alloc_ctor(0, 8, 11);
lean_ctor_set(v___x_904_, 0, v___x_899_);
lean_ctor_set(v___x_904_, 1, v___x_900_);
lean_ctor_set(v___x_904_, 2, v___x_899_);
lean_ctor_set(v___x_904_, 3, v___f_896_);
lean_ctor_set(v___x_904_, 4, v___x_901_);
lean_ctor_set(v___x_904_, 5, v___x_901_);
lean_ctor_set(v___x_904_, 6, v___x_899_);
lean_ctor_set(v___x_904_, 7, v___x_903_);
lean_ctor_set_uint8(v___x_904_, sizeof(void*)*8, v___x_850_);
lean_ctor_set_uint8(v___x_904_, sizeof(void*)*8 + 1, v___x_850_);
lean_ctor_set_uint8(v___x_904_, sizeof(void*)*8 + 2, v___x_850_);
lean_ctor_set_uint8(v___x_904_, sizeof(void*)*8 + 3, v___x_850_);
lean_ctor_set_uint8(v___x_904_, sizeof(void*)*8 + 4, v___x_902_);
lean_ctor_set_uint8(v___x_904_, sizeof(void*)*8 + 5, v___x_902_);
lean_ctor_set_uint8(v___x_904_, sizeof(void*)*8 + 6, v___x_902_);
lean_ctor_set_uint8(v___x_904_, sizeof(void*)*8 + 7, v___x_902_);
lean_ctor_set_uint8(v___x_904_, sizeof(void*)*8 + 8, v___x_850_);
lean_ctor_set_uint8(v___x_904_, sizeof(void*)*8 + 9, v___x_902_);
lean_ctor_set_uint8(v___x_904_, sizeof(void*)*8 + 10, v___x_850_);
v___x_905_ = ((lean_object*)(lp_aesop_Aesop_RuleTac_tacticMImpl___closed__3));
v___x_906_ = l_Lean_Elab_Term_TermElabM_run___redArg(v___x_898_, v___x_904_, v___x_905_, v___y_838_, v___y_839_, v___y_840_, v___y_841_);
if (lean_obj_tag(v___x_906_) == 0)
{
lean_object* v_a_907_; lean_object* v___x_908_; 
v_a_907_ = lean_ctor_get(v___x_906_, 0);
lean_inc(v_a_907_);
lean_dec_ref_known(v___x_906_, 1);
v___x_908_ = l_Lean_Meta_saveState___redArg(v___y_839_, v___y_841_);
if (lean_obj_tag(v___x_908_) == 0)
{
lean_object* v_a_909_; lean_object* v___x_910_; 
v_a_909_ = lean_ctor_get(v___x_908_, 0);
lean_inc(v_a_909_);
lean_dec_ref_known(v___x_908_, 1);
v___x_910_ = lp_aesop_Lean_getExprMVarAssignment_x3f___at___00Aesop_RuleTac_tacGenImpl_spec__0___redArg(v___x_832_, v___y_839_);
if (lean_obj_tag(v___x_910_) == 0)
{
lean_object* v_a_911_; lean_object* v_fst_912_; lean_object* v___x_913_; 
v_a_911_ = lean_ctor_get(v___x_910_, 0);
lean_inc(v_a_911_);
lean_dec_ref_known(v___x_910_, 1);
v_fst_912_ = lean_ctor_get(v_a_907_, 0);
lean_inc(v_fst_912_);
lean_dec(v_a_907_);
v___x_913_ = lean_array_mk(v_fst_912_);
if (lean_obj_tag(v_a_911_) == 1)
{
lean_object* v_val_914_; lean_object* v___x_915_; 
v_val_914_ = lean_ctor_get(v_a_911_, 0);
lean_inc(v_val_914_);
lean_dec_ref_known(v_a_911_, 1);
v___x_915_ = lp_aesop_Aesop_hasSorry___at___00Aesop_RuleTac_tacGenImpl_spec__1___redArg(v_val_914_, v___y_839_);
lean_dec(v_val_914_);
if (lean_obj_tag(v___x_915_) == 0)
{
lean_object* v_a_916_; uint8_t v___x_917_; 
v_a_916_ = lean_ctor_get(v___x_915_, 0);
lean_inc(v_a_916_);
lean_dec_ref_known(v___x_915_, 1);
v___x_917_ = lean_unbox(v_a_916_);
lean_dec(v_a_916_);
if (v___x_917_ == 0)
{
lean_object* v___x_918_; lean_object* v___x_919_; 
v___x_918_ = lean_box(0);
lean_inc(v_snd_864_);
lean_inc(v_fst_863_);
lean_inc(v___x_832_);
lean_inc_ref(v_a_831_);
v___x_919_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RuleTac_tacGenImpl_spec__2___lam__0(v_a_895_, v_a_831_, v___x_832_, v_a_909_, v___x_913_, v___x_890_, v_fst_863_, v_snd_864_, v___x_918_, v___y_837_, v___y_838_, v___y_839_, v___y_840_, v___y_841_);
v___y_883_ = v___x_919_;
goto v___jp_882_;
}
else
{
lean_object* v___x_920_; lean_object* v___x_921_; 
v___x_920_ = lean_obj_once(&lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RuleTac_tacGenImpl_spec__2___closed__2, &lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RuleTac_tacGenImpl_spec__2___closed__2_once, _init_lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RuleTac_tacGenImpl_spec__2___closed__2);
v___x_921_ = lp_aesop_Lean_throwError___at___00Lean_ofExcept___at___00Lean_evalConst___at___00Aesop_RuleTac_tacticMImpl_spec__0_spec__0_spec__1___redArg(v___x_920_, v___y_838_, v___y_839_, v___y_840_, v___y_841_);
if (lean_obj_tag(v___x_921_) == 0)
{
lean_object* v_a_922_; lean_object* v___x_923_; 
v_a_922_ = lean_ctor_get(v___x_921_, 0);
lean_inc(v_a_922_);
lean_dec_ref_known(v___x_921_, 1);
lean_inc(v_snd_864_);
lean_inc(v_fst_863_);
lean_inc(v___x_832_);
lean_inc_ref(v_a_831_);
v___x_923_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RuleTac_tacGenImpl_spec__2___lam__0(v_a_895_, v_a_831_, v___x_832_, v_a_909_, v___x_913_, v___x_890_, v_fst_863_, v_snd_864_, v_a_922_, v___y_837_, v___y_838_, v___y_839_, v___y_840_, v___y_841_);
v___y_883_ = v___x_923_;
goto v___jp_882_;
}
else
{
lean_object* v_a_924_; 
lean_dec_ref(v___x_913_);
lean_dec(v_a_909_);
lean_dec(v_a_895_);
lean_dec_ref_known(v___x_890_, 1);
v_a_924_ = lean_ctor_get(v___x_921_, 0);
lean_inc(v_a_924_);
lean_dec_ref_known(v___x_921_, 1);
v_a_879_ = v_a_924_;
goto v___jp_878_;
}
}
}
else
{
lean_object* v_a_925_; 
lean_dec_ref(v___x_913_);
lean_dec(v_a_909_);
lean_dec(v_a_895_);
lean_dec_ref_known(v___x_890_, 1);
v_a_925_ = lean_ctor_get(v___x_915_, 0);
lean_inc(v_a_925_);
lean_dec_ref_known(v___x_915_, 1);
v_a_879_ = v_a_925_;
goto v___jp_878_;
}
}
else
{
lean_object* v___x_926_; lean_object* v___x_927_; 
lean_dec(v_a_911_);
v___x_926_ = lean_box(0);
lean_inc(v_snd_864_);
lean_inc(v_fst_863_);
lean_inc(v___x_832_);
lean_inc_ref(v_a_831_);
v___x_927_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RuleTac_tacGenImpl_spec__2___lam__0(v_a_895_, v_a_831_, v___x_832_, v_a_909_, v___x_913_, v___x_890_, v_fst_863_, v_snd_864_, v___x_926_, v___y_837_, v___y_838_, v___y_839_, v___y_840_, v___y_841_);
v___y_883_ = v___x_927_;
goto v___jp_882_;
}
}
else
{
lean_object* v_a_928_; 
lean_dec(v_a_909_);
lean_dec(v_a_907_);
lean_dec(v_a_895_);
lean_dec_ref_known(v___x_890_, 1);
v_a_928_ = lean_ctor_get(v___x_910_, 0);
lean_inc(v_a_928_);
lean_dec_ref_known(v___x_910_, 1);
v_a_879_ = v_a_928_;
goto v___jp_878_;
}
}
else
{
lean_object* v_a_929_; 
lean_dec(v_a_907_);
lean_dec(v_a_895_);
lean_dec_ref_known(v___x_890_, 1);
v_a_929_ = lean_ctor_get(v___x_908_, 0);
lean_inc(v_a_929_);
lean_dec_ref_known(v___x_908_, 1);
v_a_879_ = v_a_929_;
goto v___jp_878_;
}
}
else
{
lean_object* v_a_930_; 
lean_dec(v_a_895_);
lean_dec_ref_known(v___x_890_, 1);
v_a_930_ = lean_ctor_get(v___x_906_, 0);
lean_inc(v_a_930_);
lean_dec_ref_known(v___x_906_, 1);
v_a_879_ = v_a_930_;
goto v___jp_878_;
}
}
else
{
lean_object* v___x_932_; uint8_t v_isShared_933_; uint8_t v_isSharedCheck_945_; 
lean_dec_ref(v___x_894_);
v_isSharedCheck_945_ = !lean_is_exclusive(v___x_890_);
if (v_isSharedCheck_945_ == 0)
{
lean_object* v_unused_946_; 
v_unused_946_ = lean_ctor_get(v___x_890_, 0);
lean_dec(v_unused_946_);
v___x_932_ = v___x_890_;
v_isShared_933_ = v_isSharedCheck_945_;
goto v_resetjp_931_;
}
else
{
lean_dec(v___x_890_);
v___x_932_ = lean_box(0);
v_isShared_933_ = v_isSharedCheck_945_;
goto v_resetjp_931_;
}
v_resetjp_931_:
{
lean_object* v___x_934_; lean_object* v___x_936_; 
v___x_934_ = lean_obj_once(&lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RuleTac_tacGenImpl_spec__2___closed__4, &lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RuleTac_tacGenImpl_spec__2___closed__4_once, _init_lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RuleTac_tacGenImpl_spec__2___closed__4);
lean_inc(v_fst_853_);
if (v_isShared_933_ == 0)
{
lean_ctor_set_tag(v___x_932_, 3);
lean_ctor_set(v___x_932_, 0, v_fst_853_);
v___x_936_ = v___x_932_;
goto v_reusejp_935_;
}
else
{
lean_object* v_reuseFailAlloc_944_; 
v_reuseFailAlloc_944_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v_reuseFailAlloc_944_, 0, v_fst_853_);
v___x_936_ = v_reuseFailAlloc_944_;
goto v_reusejp_935_;
}
v_reusejp_935_:
{
lean_object* v___x_937_; lean_object* v___x_938_; lean_object* v___x_940_; 
v___x_937_ = l_Lean_MessageData_ofFormat(v___x_936_);
v___x_938_ = l_Lean_indentD(v___x_937_);
if (v_isShared_857_ == 0)
{
lean_ctor_set_tag(v___x_856_, 7);
lean_ctor_set(v___x_856_, 1, v___x_938_);
lean_ctor_set(v___x_856_, 0, v___x_934_);
v___x_940_ = v___x_856_;
goto v_reusejp_939_;
}
else
{
lean_object* v_reuseFailAlloc_943_; 
v_reuseFailAlloc_943_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_943_, 0, v___x_934_);
lean_ctor_set(v_reuseFailAlloc_943_, 1, v___x_938_);
v___x_940_ = v_reuseFailAlloc_943_;
goto v_reusejp_939_;
}
v_reusejp_939_:
{
lean_object* v___x_941_; 
v___x_941_ = lp_aesop_Lean_throwError___at___00Lean_ofExcept___at___00Lean_evalConst___at___00Aesop_RuleTac_tacticMImpl_spec__0_spec__0_spec__1___redArg(v___x_940_, v___y_838_, v___y_839_, v___y_840_, v___y_841_);
if (lean_obj_tag(v___x_941_) == 0)
{
lean_dec_ref_known(v___x_941_, 1);
lean_del_object(v___x_866_);
lean_del_object(v___x_860_);
lean_dec(v_fst_853_);
v_fst_844_ = v_fst_863_;
v_snd_845_ = v_snd_864_;
goto v___jp_843_;
}
else
{
lean_object* v_a_942_; 
v_a_942_ = lean_ctor_get(v___x_941_, 0);
lean_inc(v_a_942_);
lean_dec_ref_known(v___x_941_, 1);
v_a_879_ = v_a_942_;
goto v___jp_878_;
}
}
}
}
}
}
else
{
lean_object* v___x_947_; double v___x_948_; lean_object* v___x_949_; lean_object* v___x_950_; lean_object* v___x_951_; lean_object* v___x_953_; 
lean_dec(v___x_890_);
lean_dec(v___x_862_);
v___x_947_ = lean_obj_once(&lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RuleTac_tacGenImpl_spec__2___closed__6, &lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RuleTac_tacGenImpl_spec__2___closed__6_once, _init_lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RuleTac_tacGenImpl_spec__2___closed__6);
v___x_948_ = lean_unbox_float(v_snd_854_);
lean_dec(v_snd_854_);
v___x_949_ = lean_float_to_string(v___x_948_);
v___x_950_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_950_, 0, v___x_949_);
v___x_951_ = l_Lean_MessageData_ofFormat(v___x_950_);
if (v_isShared_857_ == 0)
{
lean_ctor_set_tag(v___x_856_, 7);
lean_ctor_set(v___x_856_, 1, v___x_951_);
lean_ctor_set(v___x_856_, 0, v___x_947_);
v___x_953_ = v___x_856_;
goto v_reusejp_952_;
}
else
{
lean_object* v_reuseFailAlloc_958_; 
v_reuseFailAlloc_958_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_958_, 0, v___x_947_);
lean_ctor_set(v_reuseFailAlloc_958_, 1, v___x_951_);
v___x_953_ = v_reuseFailAlloc_958_;
goto v_reusejp_952_;
}
v_reusejp_952_:
{
lean_object* v___x_954_; lean_object* v___x_955_; lean_object* v___x_956_; 
v___x_954_ = lean_obj_once(&lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RuleTac_tacGenImpl_spec__2___closed__8, &lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RuleTac_tacGenImpl_spec__2___closed__8_once, _init_lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RuleTac_tacGenImpl_spec__2___closed__8);
v___x_955_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_955_, 0, v___x_953_);
lean_ctor_set(v___x_955_, 1, v___x_954_);
v___x_956_ = lp_aesop_Lean_throwError___at___00Lean_ofExcept___at___00Lean_evalConst___at___00Aesop_RuleTac_tacticMImpl_spec__0_spec__0_spec__1___redArg(v___x_955_, v___y_838_, v___y_839_, v___y_840_, v___y_841_);
if (lean_obj_tag(v___x_956_) == 0)
{
lean_dec_ref_known(v___x_956_, 1);
lean_del_object(v___x_866_);
lean_del_object(v___x_860_);
lean_dec(v_fst_853_);
v_fst_844_ = v_fst_863_;
v_snd_845_ = v_snd_864_;
goto v___jp_843_;
}
else
{
lean_object* v_a_957_; 
v_a_957_ = lean_ctor_get(v___x_956_, 0);
lean_inc(v_a_957_);
lean_dec_ref_known(v___x_956_, 1);
v_a_879_ = v_a_957_;
goto v___jp_878_;
}
}
}
v___jp_868_:
{
if (v___y_870_ == 0)
{
lean_object* v___x_872_; 
lean_del_object(v___x_860_);
if (v_isShared_867_ == 0)
{
lean_ctor_set(v___x_866_, 1, v___y_869_);
lean_ctor_set(v___x_866_, 0, v_fst_853_);
v___x_872_ = v___x_866_;
goto v_reusejp_871_;
}
else
{
lean_object* v_reuseFailAlloc_874_; 
v_reuseFailAlloc_874_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_874_, 0, v_fst_853_);
lean_ctor_set(v_reuseFailAlloc_874_, 1, v___y_869_);
v___x_872_ = v_reuseFailAlloc_874_;
goto v_reusejp_871_;
}
v_reusejp_871_:
{
lean_object* v___x_873_; 
v___x_873_ = lean_array_push(v_snd_864_, v___x_872_);
v_fst_844_ = v_fst_863_;
v_snd_845_ = v___x_873_;
goto v___jp_843_;
}
}
else
{
lean_object* v___x_876_; 
lean_del_object(v___x_866_);
lean_dec(v_snd_864_);
lean_dec(v_fst_863_);
lean_dec(v_fst_853_);
lean_dec(v___x_832_);
lean_dec_ref(v_a_831_);
if (v_isShared_861_ == 0)
{
lean_ctor_set_tag(v___x_860_, 1);
lean_ctor_set(v___x_860_, 0, v___y_869_);
v___x_876_ = v___x_860_;
goto v_reusejp_875_;
}
else
{
lean_object* v_reuseFailAlloc_877_; 
v_reuseFailAlloc_877_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_877_, 0, v___y_869_);
v___x_876_ = v_reuseFailAlloc_877_;
goto v_reusejp_875_;
}
v_reusejp_875_:
{
return v___x_876_;
}
}
}
v___jp_878_:
{
uint8_t v___x_880_; 
v___x_880_ = l_Lean_Exception_isInterrupt(v_a_879_);
if (v___x_880_ == 0)
{
uint8_t v___x_881_; 
lean_inc_ref(v_a_879_);
v___x_881_ = l_Lean_Exception_isRuntime(v_a_879_);
v___y_869_ = v_a_879_;
v___y_870_ = v___x_881_;
goto v___jp_868_;
}
else
{
v___y_869_ = v_a_879_;
v___y_870_ = v___x_880_;
goto v___jp_868_;
}
}
v___jp_882_:
{
if (lean_obj_tag(v___y_883_) == 0)
{
lean_object* v_a_884_; lean_object* v_snd_885_; lean_object* v_fst_886_; lean_object* v_snd_887_; 
lean_del_object(v___x_866_);
lean_dec(v_snd_864_);
lean_dec(v_fst_863_);
lean_del_object(v___x_860_);
lean_dec(v_fst_853_);
v_a_884_ = lean_ctor_get(v___y_883_, 0);
lean_inc(v_a_884_);
lean_dec_ref_known(v___y_883_, 1);
v_snd_885_ = lean_ctor_get(v_a_884_, 1);
lean_inc(v_snd_885_);
lean_dec(v_a_884_);
v_fst_886_ = lean_ctor_get(v_snd_885_, 0);
lean_inc(v_fst_886_);
v_snd_887_ = lean_ctor_get(v_snd_885_, 1);
lean_inc(v_snd_887_);
lean_dec(v_snd_885_);
v_fst_844_ = v_fst_886_;
v_snd_845_ = v_snd_887_;
goto v___jp_843_;
}
else
{
lean_object* v_a_888_; 
v_a_888_ = lean_ctor_get(v___y_883_, 0);
lean_inc(v_a_888_);
lean_dec_ref_known(v___y_883_, 1);
v_a_879_ = v_a_888_;
goto v___jp_878_;
}
}
}
}
}
else
{
lean_object* v_a_962_; lean_object* v___x_964_; uint8_t v_isShared_965_; uint8_t v_isSharedCheck_969_; 
lean_del_object(v___x_856_);
lean_dec(v_snd_854_);
lean_dec(v_fst_853_);
lean_dec_ref(v_b_836_);
lean_dec(v___x_832_);
lean_dec_ref(v_a_831_);
v_a_962_ = lean_ctor_get(v___x_858_, 0);
v_isSharedCheck_969_ = !lean_is_exclusive(v___x_858_);
if (v_isSharedCheck_969_ == 0)
{
v___x_964_ = v___x_858_;
v_isShared_965_ = v_isSharedCheck_969_;
goto v_resetjp_963_;
}
else
{
lean_inc(v_a_962_);
lean_dec(v___x_858_);
v___x_964_ = lean_box(0);
v_isShared_965_ = v_isSharedCheck_969_;
goto v_resetjp_963_;
}
v_resetjp_963_:
{
lean_object* v___x_967_; 
if (v_isShared_965_ == 0)
{
v___x_967_ = v___x_964_;
goto v_reusejp_966_;
}
else
{
lean_object* v_reuseFailAlloc_968_; 
v_reuseFailAlloc_968_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_968_, 0, v_a_962_);
v___x_967_ = v_reuseFailAlloc_968_;
goto v_reusejp_966_;
}
v_reusejp_966_:
{
return v___x_967_;
}
}
}
}
}
v___jp_843_:
{
lean_object* v___x_846_; size_t v___x_847_; size_t v___x_848_; 
v___x_846_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_846_, 0, v_fst_844_);
lean_ctor_set(v___x_846_, 1, v_snd_845_);
v___x_847_ = ((size_t)1ULL);
v___x_848_ = lean_usize_add(v_i_835_, v___x_847_);
v_i_835_ = v___x_848_;
v_b_836_ = v___x_846_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RuleTac_tacGenImpl_spec__2___boxed(lean_object* v_a_971_, lean_object* v___x_972_, lean_object* v_as_973_, lean_object* v_sz_974_, lean_object* v_i_975_, lean_object* v_b_976_, lean_object* v___y_977_, lean_object* v___y_978_, lean_object* v___y_979_, lean_object* v___y_980_, lean_object* v___y_981_, lean_object* v___y_982_){
_start:
{
size_t v_sz_boxed_983_; size_t v_i_boxed_984_; lean_object* v_res_985_; 
v_sz_boxed_983_ = lean_unbox_usize(v_sz_974_);
lean_dec(v_sz_974_);
v_i_boxed_984_ = lean_unbox_usize(v_i_975_);
lean_dec(v_i_975_);
v_res_985_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RuleTac_tacGenImpl_spec__2(v_a_971_, v___x_972_, v_as_973_, v_sz_boxed_983_, v_i_boxed_984_, v_b_976_, v___y_977_, v___y_978_, v___y_979_, v___y_980_, v___y_981_);
lean_dec(v___y_981_);
lean_dec_ref(v___y_980_);
lean_dec(v___y_979_);
lean_dec_ref(v___y_978_);
lean_dec(v___y_977_);
lean_dec_ref(v_as_973_);
return v_res_985_;
}
}
static lean_object* _init_lp_aesop_List_mapTR_loop___at___00Aesop_RuleTac_tacGenImpl_spec__3___closed__1(void){
_start:
{
lean_object* v___x_987_; lean_object* v___x_988_; 
v___x_987_ = ((lean_object*)(lp_aesop_List_mapTR_loop___at___00Aesop_RuleTac_tacGenImpl_spec__3___closed__0));
v___x_988_ = l_Lean_stringToMessageData(v___x_987_);
return v___x_988_;
}
}
LEAN_EXPORT lean_object* lp_aesop_List_mapTR_loop___at___00Aesop_RuleTac_tacGenImpl_spec__3(lean_object* v_a_989_, lean_object* v_a_990_){
_start:
{
if (lean_obj_tag(v_a_989_) == 0)
{
lean_object* v___x_991_; 
v___x_991_ = l_List_reverse___redArg(v_a_990_);
return v___x_991_;
}
else
{
lean_object* v_head_992_; lean_object* v_tail_993_; lean_object* v___x_995_; uint8_t v_isShared_996_; uint8_t v_isSharedCheck_1014_; 
v_head_992_ = lean_ctor_get(v_a_989_, 0);
v_tail_993_ = lean_ctor_get(v_a_989_, 1);
v_isSharedCheck_1014_ = !lean_is_exclusive(v_a_989_);
if (v_isSharedCheck_1014_ == 0)
{
v___x_995_ = v_a_989_;
v_isShared_996_ = v_isSharedCheck_1014_;
goto v_resetjp_994_;
}
else
{
lean_inc(v_tail_993_);
lean_inc(v_head_992_);
lean_dec(v_a_989_);
v___x_995_ = lean_box(0);
v_isShared_996_ = v_isSharedCheck_1014_;
goto v_resetjp_994_;
}
v_resetjp_994_:
{
lean_object* v_fst_997_; lean_object* v_snd_998_; lean_object* v___x_1000_; uint8_t v_isShared_1001_; uint8_t v_isSharedCheck_1013_; 
v_fst_997_ = lean_ctor_get(v_head_992_, 0);
v_snd_998_ = lean_ctor_get(v_head_992_, 1);
v_isSharedCheck_1013_ = !lean_is_exclusive(v_head_992_);
if (v_isSharedCheck_1013_ == 0)
{
v___x_1000_ = v_head_992_;
v_isShared_1001_ = v_isSharedCheck_1013_;
goto v_resetjp_999_;
}
else
{
lean_inc(v_snd_998_);
lean_inc(v_fst_997_);
lean_dec(v_head_992_);
v___x_1000_ = lean_box(0);
v_isShared_1001_ = v_isSharedCheck_1013_;
goto v_resetjp_999_;
}
v_resetjp_999_:
{
lean_object* v___x_1002_; lean_object* v___x_1003_; lean_object* v___x_1005_; 
v___x_1002_ = l_Lean_stringToMessageData(v_fst_997_);
v___x_1003_ = lean_obj_once(&lp_aesop_List_mapTR_loop___at___00Aesop_RuleTac_tacGenImpl_spec__3___closed__1, &lp_aesop_List_mapTR_loop___at___00Aesop_RuleTac_tacGenImpl_spec__3___closed__1_once, _init_lp_aesop_List_mapTR_loop___at___00Aesop_RuleTac_tacGenImpl_spec__3___closed__1);
if (v_isShared_1001_ == 0)
{
lean_ctor_set_tag(v___x_1000_, 7);
lean_ctor_set(v___x_1000_, 1, v___x_1003_);
lean_ctor_set(v___x_1000_, 0, v___x_1002_);
v___x_1005_ = v___x_1000_;
goto v_reusejp_1004_;
}
else
{
lean_object* v_reuseFailAlloc_1012_; 
v_reuseFailAlloc_1012_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1012_, 0, v___x_1002_);
lean_ctor_set(v_reuseFailAlloc_1012_, 1, v___x_1003_);
v___x_1005_ = v_reuseFailAlloc_1012_;
goto v_reusejp_1004_;
}
v_reusejp_1004_:
{
lean_object* v___x_1006_; lean_object* v___x_1007_; lean_object* v___x_1009_; 
v___x_1006_ = l_Lean_Exception_toMessageData(v_snd_998_);
v___x_1007_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1007_, 0, v___x_1005_);
lean_ctor_set(v___x_1007_, 1, v___x_1006_);
if (v_isShared_996_ == 0)
{
lean_ctor_set(v___x_995_, 1, v_a_990_);
lean_ctor_set(v___x_995_, 0, v___x_1007_);
v___x_1009_ = v___x_995_;
goto v_reusejp_1008_;
}
else
{
lean_object* v_reuseFailAlloc_1011_; 
v_reuseFailAlloc_1011_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1011_, 0, v___x_1007_);
lean_ctor_set(v_reuseFailAlloc_1011_, 1, v_a_990_);
v___x_1009_ = v_reuseFailAlloc_1011_;
goto v_reusejp_1008_;
}
v_reusejp_1008_:
{
v_a_989_ = v_tail_993_;
v_a_990_ = v___x_1009_;
goto _start;
}
}
}
}
}
}
}
static lean_object* _init_lp_aesop_Aesop_RuleTac_tacGenImpl___closed__1(void){
_start:
{
lean_object* v___x_1016_; lean_object* v___x_1017_; 
v___x_1016_ = ((lean_object*)(lp_aesop_Aesop_RuleTac_tacGenImpl___closed__0));
v___x_1017_ = l_Lean_stringToMessageData(v___x_1016_);
return v___x_1017_;
}
}
static lean_object* _init_lp_aesop_Aesop_RuleTac_tacGenImpl___closed__4(void){
_start:
{
lean_object* v___x_1021_; lean_object* v___x_1022_; 
v___x_1021_ = ((lean_object*)(lp_aesop_Aesop_RuleTac_tacGenImpl___closed__3));
v___x_1022_ = l_Lean_MessageData_ofFormat(v___x_1021_);
return v___x_1022_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTac_tacGenImpl(lean_object* v_decl_1023_, lean_object* v_input_1024_, lean_object* v_a_1025_, lean_object* v_a_1026_, lean_object* v_a_1027_, lean_object* v_a_1028_, lean_object* v_a_1029_){
_start:
{
uint8_t v___x_1031_; lean_object* v___x_1032_; 
v___x_1031_ = 1;
v___x_1032_ = lp_aesop_Lean_evalConst___at___00Aesop_RuleTac_tacticMImpl_spec__0___redArg(v_decl_1023_, v___x_1031_, v_a_1025_, v_a_1026_, v_a_1027_, v_a_1028_, v_a_1029_);
if (lean_obj_tag(v___x_1032_) == 0)
{
lean_object* v_a_1033_; lean_object* v___x_1034_; 
v_a_1033_ = lean_ctor_get(v___x_1032_, 0);
lean_inc(v_a_1033_);
lean_dec_ref_known(v___x_1032_, 1);
v___x_1034_ = l_Lean_Meta_saveState___redArg(v_a_1027_, v_a_1029_);
if (lean_obj_tag(v___x_1034_) == 0)
{
lean_object* v_a_1035_; lean_object* v_goal_1036_; lean_object* v___x_1037_; 
v_a_1035_ = lean_ctor_get(v___x_1034_, 0);
lean_inc(v_a_1035_);
lean_dec_ref_known(v___x_1034_, 1);
v_goal_1036_ = lean_ctor_get(v_input_1024_, 0);
lean_inc_n(v_goal_1036_, 2);
lean_dec_ref(v_input_1024_);
lean_inc(v_a_1029_);
lean_inc_ref(v_a_1028_);
lean_inc(v_a_1027_);
lean_inc_ref(v_a_1026_);
v___x_1037_ = lean_apply_6(v_a_1033_, v_goal_1036_, v_a_1026_, v_a_1027_, v_a_1028_, v_a_1029_, lean_box(0));
if (lean_obj_tag(v___x_1037_) == 0)
{
lean_object* v_a_1038_; lean_object* v___x_1039_; lean_object* v___x_1040_; lean_object* v___x_1041_; size_t v_sz_1042_; size_t v___x_1043_; lean_object* v___x_1044_; 
v_a_1038_ = lean_ctor_get(v___x_1037_, 0);
lean_inc(v_a_1038_);
lean_dec_ref_known(v___x_1037_, 1);
v___x_1039_ = lean_array_get_size(v_a_1038_);
v___x_1040_ = lean_mk_empty_array_with_capacity(v___x_1039_);
lean_inc_ref(v___x_1040_);
v___x_1041_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1041_, 0, v___x_1040_);
lean_ctor_set(v___x_1041_, 1, v___x_1040_);
v_sz_1042_ = lean_array_size(v_a_1038_);
v___x_1043_ = ((size_t)0ULL);
v___x_1044_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RuleTac_tacGenImpl_spec__2(v_a_1035_, v_goal_1036_, v_a_1038_, v_sz_1042_, v___x_1043_, v___x_1041_, v_a_1025_, v_a_1026_, v_a_1027_, v_a_1028_, v_a_1029_);
lean_dec(v_a_1038_);
if (lean_obj_tag(v___x_1044_) == 0)
{
lean_object* v_a_1045_; lean_object* v___x_1047_; uint8_t v_isShared_1048_; uint8_t v_isSharedCheck_1088_; 
v_a_1045_ = lean_ctor_get(v___x_1044_, 0);
v_isSharedCheck_1088_ = !lean_is_exclusive(v___x_1044_);
if (v_isSharedCheck_1088_ == 0)
{
v___x_1047_ = v___x_1044_;
v_isShared_1048_ = v_isSharedCheck_1088_;
goto v_resetjp_1046_;
}
else
{
lean_inc(v_a_1045_);
lean_dec(v___x_1044_);
v___x_1047_ = lean_box(0);
v_isShared_1048_ = v_isSharedCheck_1088_;
goto v_resetjp_1046_;
}
v_resetjp_1046_:
{
lean_object* v_fst_1049_; lean_object* v_snd_1050_; lean_object* v___x_1052_; uint8_t v_isShared_1053_; uint8_t v_isSharedCheck_1087_; 
v_fst_1049_ = lean_ctor_get(v_a_1045_, 0);
v_snd_1050_ = lean_ctor_get(v_a_1045_, 1);
v_isSharedCheck_1087_ = !lean_is_exclusive(v_a_1045_);
if (v_isSharedCheck_1087_ == 0)
{
v___x_1052_ = v_a_1045_;
v_isShared_1053_ = v_isSharedCheck_1087_;
goto v_resetjp_1051_;
}
else
{
lean_inc(v_snd_1050_);
lean_inc(v_fst_1049_);
lean_dec(v_a_1045_);
v___x_1052_ = lean_box(0);
v_isShared_1053_ = v_isSharedCheck_1087_;
goto v_resetjp_1051_;
}
v_resetjp_1051_:
{
lean_object* v___x_1054_; lean_object* v___x_1055_; uint8_t v___x_1056_; 
v___x_1054_ = lean_array_get_size(v_fst_1049_);
v___x_1055_ = lean_unsigned_to_nat(0u);
v___x_1056_ = lean_nat_dec_eq(v___x_1054_, v___x_1055_);
if (v___x_1056_ == 0)
{
lean_object* v___x_1058_; 
lean_del_object(v___x_1052_);
lean_dec(v_snd_1050_);
if (v_isShared_1048_ == 0)
{
lean_ctor_set(v___x_1047_, 0, v_fst_1049_);
v___x_1058_ = v___x_1047_;
goto v_reusejp_1057_;
}
else
{
lean_object* v_reuseFailAlloc_1059_; 
v_reuseFailAlloc_1059_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1059_, 0, v_fst_1049_);
v___x_1058_ = v_reuseFailAlloc_1059_;
goto v_reusejp_1057_;
}
v_reusejp_1057_:
{
return v___x_1058_;
}
}
else
{
lean_object* v___x_1060_; lean_object* v___x_1061_; lean_object* v___x_1062_; lean_object* v___x_1063_; lean_object* v___x_1064_; lean_object* v___x_1065_; lean_object* v___x_1066_; lean_object* v___x_1068_; 
lean_del_object(v___x_1047_);
v___x_1060_ = lean_obj_once(&lp_aesop_Aesop_RuleTac_tacGenImpl___closed__1, &lp_aesop_Aesop_RuleTac_tacGenImpl___closed__1_once, _init_lp_aesop_Aesop_RuleTac_tacGenImpl___closed__1);
v___x_1061_ = lean_array_to_list(v_snd_1050_);
v___x_1062_ = lean_box(0);
v___x_1063_ = lp_aesop_List_mapTR_loop___at___00Aesop_RuleTac_tacGenImpl_spec__3(v___x_1061_, v___x_1062_);
v___x_1064_ = lean_obj_once(&lp_aesop_Aesop_RuleTac_tacGenImpl___closed__4, &lp_aesop_Aesop_RuleTac_tacGenImpl___closed__4_once, _init_lp_aesop_Aesop_RuleTac_tacGenImpl___closed__4);
v___x_1065_ = l_Lean_MessageData_joinSep(v___x_1063_, v___x_1064_);
v___x_1066_ = l_Lean_indentD(v___x_1065_);
if (v_isShared_1053_ == 0)
{
lean_ctor_set_tag(v___x_1052_, 7);
lean_ctor_set(v___x_1052_, 1, v___x_1066_);
lean_ctor_set(v___x_1052_, 0, v___x_1060_);
v___x_1068_ = v___x_1052_;
goto v_reusejp_1067_;
}
else
{
lean_object* v_reuseFailAlloc_1086_; 
v_reuseFailAlloc_1086_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1086_, 0, v___x_1060_);
lean_ctor_set(v_reuseFailAlloc_1086_, 1, v___x_1066_);
v___x_1068_ = v_reuseFailAlloc_1086_;
goto v_reusejp_1067_;
}
v_reusejp_1067_:
{
lean_object* v___x_1069_; 
v___x_1069_ = lp_aesop_Lean_throwError___at___00Lean_ofExcept___at___00Lean_evalConst___at___00Aesop_RuleTac_tacticMImpl_spec__0_spec__0_spec__1___redArg(v___x_1068_, v_a_1026_, v_a_1027_, v_a_1028_, v_a_1029_);
if (lean_obj_tag(v___x_1069_) == 0)
{
lean_object* v___x_1071_; uint8_t v_isShared_1072_; uint8_t v_isSharedCheck_1076_; 
v_isSharedCheck_1076_ = !lean_is_exclusive(v___x_1069_);
if (v_isSharedCheck_1076_ == 0)
{
lean_object* v_unused_1077_; 
v_unused_1077_ = lean_ctor_get(v___x_1069_, 0);
lean_dec(v_unused_1077_);
v___x_1071_ = v___x_1069_;
v_isShared_1072_ = v_isSharedCheck_1076_;
goto v_resetjp_1070_;
}
else
{
lean_dec(v___x_1069_);
v___x_1071_ = lean_box(0);
v_isShared_1072_ = v_isSharedCheck_1076_;
goto v_resetjp_1070_;
}
v_resetjp_1070_:
{
lean_object* v___x_1074_; 
if (v_isShared_1072_ == 0)
{
lean_ctor_set(v___x_1071_, 0, v_fst_1049_);
v___x_1074_ = v___x_1071_;
goto v_reusejp_1073_;
}
else
{
lean_object* v_reuseFailAlloc_1075_; 
v_reuseFailAlloc_1075_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1075_, 0, v_fst_1049_);
v___x_1074_ = v_reuseFailAlloc_1075_;
goto v_reusejp_1073_;
}
v_reusejp_1073_:
{
return v___x_1074_;
}
}
}
else
{
lean_object* v_a_1078_; lean_object* v___x_1080_; uint8_t v_isShared_1081_; uint8_t v_isSharedCheck_1085_; 
lean_dec(v_fst_1049_);
v_a_1078_ = lean_ctor_get(v___x_1069_, 0);
v_isSharedCheck_1085_ = !lean_is_exclusive(v___x_1069_);
if (v_isSharedCheck_1085_ == 0)
{
v___x_1080_ = v___x_1069_;
v_isShared_1081_ = v_isSharedCheck_1085_;
goto v_resetjp_1079_;
}
else
{
lean_inc(v_a_1078_);
lean_dec(v___x_1069_);
v___x_1080_ = lean_box(0);
v_isShared_1081_ = v_isSharedCheck_1085_;
goto v_resetjp_1079_;
}
v_resetjp_1079_:
{
lean_object* v___x_1083_; 
if (v_isShared_1081_ == 0)
{
v___x_1083_ = v___x_1080_;
goto v_reusejp_1082_;
}
else
{
lean_object* v_reuseFailAlloc_1084_; 
v_reuseFailAlloc_1084_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1084_, 0, v_a_1078_);
v___x_1083_ = v_reuseFailAlloc_1084_;
goto v_reusejp_1082_;
}
v_reusejp_1082_:
{
return v___x_1083_;
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
lean_object* v_a_1089_; lean_object* v___x_1091_; uint8_t v_isShared_1092_; uint8_t v_isSharedCheck_1096_; 
v_a_1089_ = lean_ctor_get(v___x_1044_, 0);
v_isSharedCheck_1096_ = !lean_is_exclusive(v___x_1044_);
if (v_isSharedCheck_1096_ == 0)
{
v___x_1091_ = v___x_1044_;
v_isShared_1092_ = v_isSharedCheck_1096_;
goto v_resetjp_1090_;
}
else
{
lean_inc(v_a_1089_);
lean_dec(v___x_1044_);
v___x_1091_ = lean_box(0);
v_isShared_1092_ = v_isSharedCheck_1096_;
goto v_resetjp_1090_;
}
v_resetjp_1090_:
{
lean_object* v___x_1094_; 
if (v_isShared_1092_ == 0)
{
v___x_1094_ = v___x_1091_;
goto v_reusejp_1093_;
}
else
{
lean_object* v_reuseFailAlloc_1095_; 
v_reuseFailAlloc_1095_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1095_, 0, v_a_1089_);
v___x_1094_ = v_reuseFailAlloc_1095_;
goto v_reusejp_1093_;
}
v_reusejp_1093_:
{
return v___x_1094_;
}
}
}
}
else
{
lean_object* v_a_1097_; lean_object* v___x_1099_; uint8_t v_isShared_1100_; uint8_t v_isSharedCheck_1104_; 
lean_dec(v_goal_1036_);
lean_dec(v_a_1035_);
v_a_1097_ = lean_ctor_get(v___x_1037_, 0);
v_isSharedCheck_1104_ = !lean_is_exclusive(v___x_1037_);
if (v_isSharedCheck_1104_ == 0)
{
v___x_1099_ = v___x_1037_;
v_isShared_1100_ = v_isSharedCheck_1104_;
goto v_resetjp_1098_;
}
else
{
lean_inc(v_a_1097_);
lean_dec(v___x_1037_);
v___x_1099_ = lean_box(0);
v_isShared_1100_ = v_isSharedCheck_1104_;
goto v_resetjp_1098_;
}
v_resetjp_1098_:
{
lean_object* v___x_1102_; 
if (v_isShared_1100_ == 0)
{
v___x_1102_ = v___x_1099_;
goto v_reusejp_1101_;
}
else
{
lean_object* v_reuseFailAlloc_1103_; 
v_reuseFailAlloc_1103_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1103_, 0, v_a_1097_);
v___x_1102_ = v_reuseFailAlloc_1103_;
goto v_reusejp_1101_;
}
v_reusejp_1101_:
{
return v___x_1102_;
}
}
}
}
else
{
lean_object* v_a_1105_; lean_object* v___x_1107_; uint8_t v_isShared_1108_; uint8_t v_isSharedCheck_1112_; 
lean_dec(v_a_1033_);
lean_dec_ref(v_input_1024_);
v_a_1105_ = lean_ctor_get(v___x_1034_, 0);
v_isSharedCheck_1112_ = !lean_is_exclusive(v___x_1034_);
if (v_isSharedCheck_1112_ == 0)
{
v___x_1107_ = v___x_1034_;
v_isShared_1108_ = v_isSharedCheck_1112_;
goto v_resetjp_1106_;
}
else
{
lean_inc(v_a_1105_);
lean_dec(v___x_1034_);
v___x_1107_ = lean_box(0);
v_isShared_1108_ = v_isSharedCheck_1112_;
goto v_resetjp_1106_;
}
v_resetjp_1106_:
{
lean_object* v___x_1110_; 
if (v_isShared_1108_ == 0)
{
v___x_1110_ = v___x_1107_;
goto v_reusejp_1109_;
}
else
{
lean_object* v_reuseFailAlloc_1111_; 
v_reuseFailAlloc_1111_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1111_, 0, v_a_1105_);
v___x_1110_ = v_reuseFailAlloc_1111_;
goto v_reusejp_1109_;
}
v_reusejp_1109_:
{
return v___x_1110_;
}
}
}
}
else
{
lean_object* v_a_1113_; lean_object* v___x_1115_; uint8_t v_isShared_1116_; uint8_t v_isSharedCheck_1120_; 
lean_dec_ref(v_input_1024_);
v_a_1113_ = lean_ctor_get(v___x_1032_, 0);
v_isSharedCheck_1120_ = !lean_is_exclusive(v___x_1032_);
if (v_isSharedCheck_1120_ == 0)
{
v___x_1115_ = v___x_1032_;
v_isShared_1116_ = v_isSharedCheck_1120_;
goto v_resetjp_1114_;
}
else
{
lean_inc(v_a_1113_);
lean_dec(v___x_1032_);
v___x_1115_ = lean_box(0);
v_isShared_1116_ = v_isSharedCheck_1120_;
goto v_resetjp_1114_;
}
v_resetjp_1114_:
{
lean_object* v___x_1118_; 
if (v_isShared_1116_ == 0)
{
v___x_1118_ = v___x_1115_;
goto v_reusejp_1117_;
}
else
{
lean_object* v_reuseFailAlloc_1119_; 
v_reuseFailAlloc_1119_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1119_, 0, v_a_1113_);
v___x_1118_ = v_reuseFailAlloc_1119_;
goto v_reusejp_1117_;
}
v_reusejp_1117_:
{
return v___x_1118_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTac_tacGenImpl___boxed(lean_object* v_decl_1121_, lean_object* v_input_1122_, lean_object* v_a_1123_, lean_object* v_a_1124_, lean_object* v_a_1125_, lean_object* v_a_1126_, lean_object* v_a_1127_, lean_object* v_a_1128_){
_start:
{
lean_object* v_res_1129_; 
v_res_1129_ = lp_aesop_Aesop_RuleTac_tacGenImpl(v_decl_1121_, v_input_1122_, v_a_1123_, v_a_1124_, v_a_1125_, v_a_1126_, v_a_1127_);
lean_dec(v_a_1127_);
lean_dec_ref(v_a_1126_);
lean_dec(v_a_1125_);
lean_dec_ref(v_a_1124_);
lean_dec(v_a_1123_);
return v_res_1129_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_RuleTac_Basic(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_aesop_Aesop_RuleTac_Tactic(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_RuleTac_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_aesop_Aesop_RuleTac_Tactic(uint8_t builtin) {
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
lean_object* initialize_aesop_Aesop_RuleTac_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_aesop_Aesop_RuleTac_Tactic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_RuleTac_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_RuleTac_Tactic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_aesop_Aesop_RuleTac_Tactic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_aesop_Aesop_RuleTac_Tactic(builtin);
}
#ifdef __cplusplus
}
#endif
