// Lean compiler output
// Module: Mathlib.Tactic.SimpIntro
// Imports: public import Init public meta import Init public meta import Lean.Elab.Tactic.Simp public import Mathlib.Init public import Lean.Elab.Tactic.Simp
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
lean_object* l_Array_mkArray0(lean_object*);
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
lean_object* lean_st_ref_get(lean_object*);
lean_object* l_Lean_Elab_getBetterRef(lean_object*, lean_object*);
extern lean_object* l_Lean_Elab_pp_macroStack;
lean_object* l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(lean_object*, lean_object*);
lean_object* l_Lean_MessageData_ofFormat(lean_object*);
lean_object* l_Lean_MessageData_ofSyntax(lean_object*);
lean_object* l_Lean_indentD(lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
extern lean_object* l_Lean_Elab_unsupportedSyntaxExceptionId;
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_withMVarContextImp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_Parser_Tactic_simpArgs;
extern lean_object* l_Lean_binderIdent;
extern lean_object* l_Lean_Parser_Tactic_discharger;
extern lean_object* l_Lean_Parser_Tactic_optConfig;
lean_object* l_Lean_Name_mkStr3(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* l_Array_append___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node6(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_getSimpTheorems___boxed(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_mkSimpContext___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_withMainContext___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_getMainGoal___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MVarId_checkNotAssigned(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_to_list(lean_object*);
lean_object* l_Lean_mkFVar(lean_object*);
lean_object* l_Lean_Elab_Term_addLocalVarInfo(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_FVarId_getType___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_isProp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_fvar___override(lean_object*);
extern lean_object* l_Lean_Meta_simpGlobalConfig;
lean_object* l_Lean_Meta_SimpTheoremsArray_addTheorem(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_Simp_Context_setSimpTheorems(lean_object*, lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* l_Lean_MVarId_intro(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Expr_hasLooseBVars(lean_object*);
lean_object* l_Lean_Meta_simpLocalDecl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Term_addLocalVarInfo___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_List_isEmpty___redArg(lean_object*);
lean_object* l_Lean_Meta_simpTargetCore(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isIdent(lean_object*);
lean_object* l_Lean_Syntax_getId(lean_object*);
lean_object* l_Lean_Meta_ConfigWithKey_setTransparency(uint8_t, lean_object*);
lean_object* l_Lean_MVarId_getType_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_mkHole(lean_object*, uint8_t);
lean_object* l_Lean_Elab_Tactic_replaceMainGoal___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_Simp_DischargeWrapper_with___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Array_mkArray3___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Array_mkArray1___redArg(lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArgs(lean_object*);
lean_object* l_Lean_Syntax_getOptional_x3f(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_simpIntroCore_spec__0___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_simpIntroCore_spec__0___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_simpIntroCore_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_simpIntroCore_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_simpIntroCore_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_simpIntroCore_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_simpIntroCore_spec__1_spec__1_spec__3_spec__5___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_simpIntroCore_spec__1_spec__1_spec__3_spec__5___closed__0;
static const lean_string_object lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_simpIntroCore_spec__1_spec__1_spec__3_spec__5___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "while expanding"};
static const lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_simpIntroCore_spec__1_spec__1_spec__3_spec__5___closed__1 = (const lean_object*)&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_simpIntroCore_spec__1_spec__1_spec__3_spec__5___closed__1_value;
static const lean_ctor_object lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_simpIntroCore_spec__1_spec__1_spec__3_spec__5___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_simpIntroCore_spec__1_spec__1_spec__3_spec__5___closed__1_value)}};
static const lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_simpIntroCore_spec__1_spec__1_spec__3_spec__5___closed__2 = (const lean_object*)&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_simpIntroCore_spec__1_spec__1_spec__3_spec__5___closed__2_value;
static lean_once_cell_t lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_simpIntroCore_spec__1_spec__1_spec__3_spec__5___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_simpIntroCore_spec__1_spec__1_spec__3_spec__5___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_simpIntroCore_spec__1_spec__1_spec__3_spec__5(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_simpIntroCore_spec__1_spec__1_spec__3_spec__4(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_simpIntroCore_spec__1_spec__1_spec__3_spec__4___boxed(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_simpIntroCore_spec__1_spec__1_spec__3___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 25, .m_capacity = 25, .m_length = 24, .m_data = "with resulting expansion"};
static const lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_simpIntroCore_spec__1_spec__1_spec__3___redArg___closed__0 = (const lean_object*)&lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_simpIntroCore_spec__1_spec__1_spec__3___redArg___closed__0_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_simpIntroCore_spec__1_spec__1_spec__3___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_simpIntroCore_spec__1_spec__1_spec__3___redArg___closed__0_value)}};
static const lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_simpIntroCore_spec__1_spec__1_spec__3___redArg___closed__1 = (const lean_object*)&lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_simpIntroCore_spec__1_spec__1_spec__3___redArg___closed__1_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_simpIntroCore_spec__1_spec__1_spec__3___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_simpIntroCore_spec__1_spec__1_spec__3___redArg___closed__2;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_simpIntroCore_spec__1_spec__1_spec__3___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_simpIntroCore_spec__1_spec__1_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_simpIntroCore_spec__1_spec__1_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_simpIntroCore_spec__1_spec__1_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_simpIntroCore_spec__1_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_simpIntroCore_spec__1_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Mathlib_Tactic_simpIntroCore_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Mathlib_Tactic_simpIntroCore_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_simpIntroCore___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_simpIntroCore___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 32, .m_capacity = 32, .m_length = 31, .m_data = "simp_intro failed to introduce "};
static const lean_object* lp_mathlib_Mathlib_Tactic_simpIntroCore___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_simpIntroCore___closed__0_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_simpIntroCore___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_simpIntroCore___closed__1;
static const lean_string_object lp_mathlib_Mathlib_Tactic_simpIntroCore___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "\n"};
static const lean_object* lp_mathlib_Mathlib_Tactic_simpIntroCore___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_simpIntroCore___closed__2_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_simpIntroCore___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_simpIntroCore___closed__3;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_simpIntroCore___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_simpIntroCore___closed__7;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_simpIntroCore___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_simpIntroCore___closed__8;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_simpIntroCore___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_simpIntroCore___closed__4;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_simpIntroCore___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_simpIntroCore___closed__5;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_simpIntroCore___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_simpIntroCore___closed__9;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_simpIntroCore___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_simpIntroCore___closed__6;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_simpIntroCore___closed__10_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_simpIntroCore___closed__10;
static const lean_string_object lp_mathlib_Mathlib_Tactic_simpIntroCore___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "_"};
static const lean_object* lp_mathlib_Mathlib_Tactic_simpIntroCore___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_simpIntroCore___closed__11_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_simpIntroCore___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_simpIntroCore___closed__11_value),LEAN_SCALAR_PTR_LITERAL(168, 60, 211, 188, 58, 220, 100, 184)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_simpIntroCore___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_simpIntroCore___closed__12_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_simpIntroCore(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_simpIntroCore___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_simpIntroCore___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Mathlib_Tactic_simpIntroCore_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Mathlib_Tactic_simpIntroCore_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_simpIntroCore_spec__1_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_simpIntroCore_spec__1_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_simpIntroCore_spec__1_spec__1_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_simpIntroCore_spec__1_spec__1_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 29, .m_capacity = 29, .m_length = 28, .m_data = "tacticSimp_intro_____..Only_"};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__3_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__3_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__2_value),LEAN_SCALAR_PTR_LITERAL(131, 11, 25, 248, 221, 64, 181, 114)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__4_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "simp_intro"};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__6_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__7_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__8;
static const lean_string_object lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "optional"};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__9_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__9_value),LEAN_SCALAR_PTR_LITERAL(233, 141, 154, 50, 143, 135, 42, 252)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__10_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__11_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__11;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__12_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__12;
static const lean_string_object lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "many"};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__13_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__13_value),LEAN_SCALAR_PTR_LITERAL(41, 35, 40, 86, 189, 97, 244, 31)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__14 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__14_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "ppSpace"};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__15 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__15_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__15_value),LEAN_SCALAR_PTR_LITERAL(207, 47, 58, 43, 30, 240, 125, 246)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__16 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__16_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__16_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__17 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__17_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "colGt"};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__18 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__18_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__18_value),LEAN_SCALAR_PTR_LITERAL(185, 236, 32, 153, 169, 213, 53, 244)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__19 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__19_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__19_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__20 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__20_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__17_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__20_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__21 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__21_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__22_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__22;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__23_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__23;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__24_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__24;
static const lean_string_object lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = " .."};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__25 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__25_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__26_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__25_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__26 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__26_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__27_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__10_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__26_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__27 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__27_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__28_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__28;
static const lean_string_object lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__29_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = " only"};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__29 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__29_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__30_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__29_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__30 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__30_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__31_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__10_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__30_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__31 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__31_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__32_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__32;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__33_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__33;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__34_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__34;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__35_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__35;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly__;
static lean_once_cell_t lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SimpIntro______elabRules__Mathlib__Tactic__tacticSimp__intro___________x2e_x2eOnly____1_spec__0___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SimpIntro______elabRules__Mathlib__Tactic__tacticSimp__intro___________x2e_x2eOnly____1_spec__0___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SimpIntro______elabRules__Mathlib__Tactic__tacticSimp__intro___________x2e_x2eOnly____1_spec__0___redArg();
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SimpIntro______elabRules__Mathlib__Tactic__tacticSimp__intro___________x2e_x2eOnly____1_spec__0___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SimpIntro______elabRules__Mathlib__Tactic__tacticSimp__intro___________x2e_x2eOnly____1_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SimpIntro______elabRules__Mathlib__Tactic__tacticSimp__intro___________x2e_x2eOnly____1_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SimpIntro______elabRules__Mathlib__Tactic__tacticSimp__intro___________x2e_x2eOnly____1_spec__1___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SimpIntro______elabRules__Mathlib__Tactic__tacticSimp__intro___________x2e_x2eOnly____1_spec__1___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SimpIntro______elabRules__Mathlib__Tactic__tacticSimp__intro___________x2e_x2eOnly____1_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SimpIntro______elabRules__Mathlib__Tactic__tacticSimp__intro___________x2e_x2eOnly____1_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SimpIntro______elabRules__Mathlib__Tactic__tacticSimp__intro___________x2e_x2eOnly____1_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SimpIntro______elabRules__Mathlib__Tactic__tacticSimp__intro___________x2e_x2eOnly____1_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpIntro______elabRules__Mathlib__Tactic__tacticSimp__intro___________x2e_x2eOnly____1___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpIntro______elabRules__Mathlib__Tactic__tacticSimp__intro___________x2e_x2eOnly____1___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpIntro______elabRules__Mathlib__Tactic__tacticSimp__intro___________x2e_x2eOnly____1___lam__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__6_value),LEAN_SCALAR_PTR_LITERAL(152, 54, 108, 95, 216, 211, 60, 230)}};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpIntro______elabRules__Mathlib__Tactic__tacticSimp__intro___________x2e_x2eOnly____1___lam__1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpIntro______elabRules__Mathlib__Tactic__tacticSimp__intro___________x2e_x2eOnly____1___lam__1___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpIntro______elabRules__Mathlib__Tactic__tacticSimp__intro___________x2e_x2eOnly____1___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpIntro______elabRules__Mathlib__Tactic__tacticSimp__intro___________x2e_x2eOnly____1___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpIntro______elabRules__Mathlib__Tactic__tacticSimp__intro___________x2e_x2eOnly____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Meta_getSimpTheorems___boxed, .m_arity = 3, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpIntro______elabRules__Mathlib__Tactic__tacticSimp__intro___________x2e_x2eOnly____1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpIntro______elabRules__Mathlib__Tactic__tacticSimp__intro___________x2e_x2eOnly____1___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpIntro______elabRules__Mathlib__Tactic__tacticSimp__intro___________x2e_x2eOnly____1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "["};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpIntro______elabRules__Mathlib__Tactic__tacticSimp__intro___________x2e_x2eOnly____1___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpIntro______elabRules__Mathlib__Tactic__tacticSimp__intro___________x2e_x2eOnly____1___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpIntro______elabRules__Mathlib__Tactic__tacticSimp__intro___________x2e_x2eOnly____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "]"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpIntro______elabRules__Mathlib__Tactic__tacticSimp__intro___________x2e_x2eOnly____1___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpIntro______elabRules__Mathlib__Tactic__tacticSimp__intro___________x2e_x2eOnly____1___closed__2_value;
static const lean_array_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpIntro______elabRules__Mathlib__Tactic__tacticSimp__intro___________x2e_x2eOnly____1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpIntro______elabRules__Mathlib__Tactic__tacticSimp__intro___________x2e_x2eOnly____1___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpIntro______elabRules__Mathlib__Tactic__tacticSimp__intro___________x2e_x2eOnly____1___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpIntro______elabRules__Mathlib__Tactic__tacticSimp__intro___________x2e_x2eOnly____1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "only"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpIntro______elabRules__Mathlib__Tactic__tacticSimp__intro___________x2e_x2eOnly____1___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpIntro______elabRules__Mathlib__Tactic__tacticSimp__intro___________x2e_x2eOnly____1___closed__4_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpIntro______elabRules__Mathlib__Tactic__tacticSimp__intro___________x2e_x2eOnly____1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "simp"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpIntro______elabRules__Mathlib__Tactic__tacticSimp__intro___________x2e_x2eOnly____1___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpIntro______elabRules__Mathlib__Tactic__tacticSimp__intro___________x2e_x2eOnly____1___closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpIntro______elabRules__Mathlib__Tactic__tacticSimp__intro___________x2e_x2eOnly____1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpIntro______elabRules__Mathlib__Tactic__tacticSimp__intro___________x2e_x2eOnly____1___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpIntro______elabRules__Mathlib__Tactic__tacticSimp__intro___________x2e_x2eOnly____1___closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpIntro______elabRules__Mathlib__Tactic__tacticSimp__intro___________x2e_x2eOnly____1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpIntro______elabRules__Mathlib__Tactic__tacticSimp__intro___________x2e_x2eOnly____1___closed__6_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpIntro______elabRules__Mathlib__Tactic__tacticSimp__intro___________x2e_x2eOnly____1___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpIntro______elabRules__Mathlib__Tactic__tacticSimp__intro___________x2e_x2eOnly____1___closed__7_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpIntro______elabRules__Mathlib__Tactic__tacticSimp__intro___________x2e_x2eOnly____1___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpIntro______elabRules__Mathlib__Tactic__tacticSimp__intro___________x2e_x2eOnly____1___closed__8;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpIntro______elabRules__Mathlib__Tactic__tacticSimp__intro___________x2e_x2eOnly____1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpIntro______elabRules__Mathlib__Tactic__tacticSimp__intro___________x2e_x2eOnly____1___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpIntro______elabRules__Mathlib__Tactic__tacticSimp__intro___________x2e_x2eOnly____1___closed__9_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpIntro______elabRules__Mathlib__Tactic__tacticSimp__intro___________x2e_x2eOnly____1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpIntro______elabRules__Mathlib__Tactic__tacticSimp__intro___________x2e_x2eOnly____1___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpIntro______elabRules__Mathlib__Tactic__tacticSimp__intro___________x2e_x2eOnly____1___closed__10_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpIntro______elabRules__Mathlib__Tactic__tacticSimp__intro___________x2e_x2eOnly____1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpIntro______elabRules__Mathlib__Tactic__tacticSimp__intro___________x2e_x2eOnly____1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_simpIntroCore_spec__0___redArg___lam__0(lean_object* v_x_1_, lean_object* v___y_2_, lean_object* v___y_3_, lean_object* v___y_4_, lean_object* v___y_5_, lean_object* v___y_6_, lean_object* v___y_7_){
_start:
{
lean_object* v___x_9_; 
lean_inc(v___y_3_);
lean_inc_ref(v___y_2_);
v___x_9_ = lean_apply_7(v_x_1_, v___y_2_, v___y_3_, v___y_4_, v___y_5_, v___y_6_, v___y_7_, lean_box(0));
return v___x_9_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_simpIntroCore_spec__0___redArg___lam__0___boxed(lean_object* v_x_10_, lean_object* v___y_11_, lean_object* v___y_12_, lean_object* v___y_13_, lean_object* v___y_14_, lean_object* v___y_15_, lean_object* v___y_16_, lean_object* v___y_17_){
_start:
{
lean_object* v_res_18_; 
v_res_18_ = lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_simpIntroCore_spec__0___redArg___lam__0(v_x_10_, v___y_11_, v___y_12_, v___y_13_, v___y_14_, v___y_15_, v___y_16_);
lean_dec(v___y_12_);
lean_dec_ref(v___y_11_);
return v_res_18_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_simpIntroCore_spec__0___redArg(lean_object* v_mvarId_19_, lean_object* v_x_20_, lean_object* v___y_21_, lean_object* v___y_22_, lean_object* v___y_23_, lean_object* v___y_24_, lean_object* v___y_25_, lean_object* v___y_26_){
_start:
{
lean_object* v___f_28_; lean_object* v___x_29_; 
lean_inc(v___y_22_);
lean_inc_ref(v___y_21_);
v___f_28_ = lean_alloc_closure((void*)(lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_simpIntroCore_spec__0___redArg___lam__0___boxed), 8, 3);
lean_closure_set(v___f_28_, 0, v_x_20_);
lean_closure_set(v___f_28_, 1, v___y_21_);
lean_closure_set(v___f_28_, 2, v___y_22_);
v___x_29_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withMVarContextImp(lean_box(0), v_mvarId_19_, v___f_28_, v___y_23_, v___y_24_, v___y_25_, v___y_26_);
if (lean_obj_tag(v___x_29_) == 0)
{
return v___x_29_;
}
else
{
lean_object* v_a_30_; lean_object* v___x_32_; uint8_t v_isShared_33_; uint8_t v_isSharedCheck_37_; 
v_a_30_ = lean_ctor_get(v___x_29_, 0);
v_isSharedCheck_37_ = !lean_is_exclusive(v___x_29_);
if (v_isSharedCheck_37_ == 0)
{
v___x_32_ = v___x_29_;
v_isShared_33_ = v_isSharedCheck_37_;
goto v_resetjp_31_;
}
else
{
lean_inc(v_a_30_);
lean_dec(v___x_29_);
v___x_32_ = lean_box(0);
v_isShared_33_ = v_isSharedCheck_37_;
goto v_resetjp_31_;
}
v_resetjp_31_:
{
lean_object* v___x_35_; 
if (v_isShared_33_ == 0)
{
v___x_35_ = v___x_32_;
goto v_reusejp_34_;
}
else
{
lean_object* v_reuseFailAlloc_36_; 
v_reuseFailAlloc_36_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_36_, 0, v_a_30_);
v___x_35_ = v_reuseFailAlloc_36_;
goto v_reusejp_34_;
}
v_reusejp_34_:
{
return v___x_35_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_simpIntroCore_spec__0___redArg___boxed(lean_object* v_mvarId_38_, lean_object* v_x_39_, lean_object* v___y_40_, lean_object* v___y_41_, lean_object* v___y_42_, lean_object* v___y_43_, lean_object* v___y_44_, lean_object* v___y_45_, lean_object* v___y_46_){
_start:
{
lean_object* v_res_47_; 
v_res_47_ = lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_simpIntroCore_spec__0___redArg(v_mvarId_38_, v_x_39_, v___y_40_, v___y_41_, v___y_42_, v___y_43_, v___y_44_, v___y_45_);
lean_dec(v___y_45_);
lean_dec_ref(v___y_44_);
lean_dec(v___y_43_);
lean_dec_ref(v___y_42_);
lean_dec(v___y_41_);
lean_dec_ref(v___y_40_);
return v_res_47_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_simpIntroCore_spec__0(lean_object* v_00_u03b1_48_, lean_object* v_mvarId_49_, lean_object* v_x_50_, lean_object* v___y_51_, lean_object* v___y_52_, lean_object* v___y_53_, lean_object* v___y_54_, lean_object* v___y_55_, lean_object* v___y_56_){
_start:
{
lean_object* v___x_58_; 
v___x_58_ = lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_simpIntroCore_spec__0___redArg(v_mvarId_49_, v_x_50_, v___y_51_, v___y_52_, v___y_53_, v___y_54_, v___y_55_, v___y_56_);
return v___x_58_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_simpIntroCore_spec__0___boxed(lean_object* v_00_u03b1_59_, lean_object* v_mvarId_60_, lean_object* v_x_61_, lean_object* v___y_62_, lean_object* v___y_63_, lean_object* v___y_64_, lean_object* v___y_65_, lean_object* v___y_66_, lean_object* v___y_67_, lean_object* v___y_68_){
_start:
{
lean_object* v_res_69_; 
v_res_69_ = lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_simpIntroCore_spec__0(v_00_u03b1_59_, v_mvarId_60_, v_x_61_, v___y_62_, v___y_63_, v___y_64_, v___y_65_, v___y_66_, v___y_67_);
lean_dec(v___y_67_);
lean_dec_ref(v___y_66_);
lean_dec(v___y_65_);
lean_dec_ref(v___y_64_);
lean_dec(v___y_63_);
lean_dec_ref(v___y_62_);
return v_res_69_;
}
}
static lean_object* _init_lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_simpIntroCore_spec__1_spec__1_spec__3_spec__5___closed__0(void){
_start:
{
lean_object* v___x_70_; lean_object* v___x_71_; 
v___x_70_ = lean_box(1);
v___x_71_ = l_Lean_MessageData_ofFormat(v___x_70_);
return v___x_71_;
}
}
static lean_object* _init_lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_simpIntroCore_spec__1_spec__1_spec__3_spec__5___closed__3(void){
_start:
{
lean_object* v___x_75_; lean_object* v___x_76_; 
v___x_75_ = ((lean_object*)(lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_simpIntroCore_spec__1_spec__1_spec__3_spec__5___closed__2));
v___x_76_ = l_Lean_MessageData_ofFormat(v___x_75_);
return v___x_76_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_simpIntroCore_spec__1_spec__1_spec__3_spec__5(lean_object* v_x_77_, lean_object* v_x_78_){
_start:
{
if (lean_obj_tag(v_x_78_) == 0)
{
return v_x_77_;
}
else
{
lean_object* v_head_79_; lean_object* v_tail_80_; lean_object* v___x_82_; uint8_t v_isShared_83_; uint8_t v_isSharedCheck_102_; 
v_head_79_ = lean_ctor_get(v_x_78_, 0);
v_tail_80_ = lean_ctor_get(v_x_78_, 1);
v_isSharedCheck_102_ = !lean_is_exclusive(v_x_78_);
if (v_isSharedCheck_102_ == 0)
{
v___x_82_ = v_x_78_;
v_isShared_83_ = v_isSharedCheck_102_;
goto v_resetjp_81_;
}
else
{
lean_inc(v_tail_80_);
lean_inc(v_head_79_);
lean_dec(v_x_78_);
v___x_82_ = lean_box(0);
v_isShared_83_ = v_isSharedCheck_102_;
goto v_resetjp_81_;
}
v_resetjp_81_:
{
lean_object* v_before_84_; lean_object* v___x_86_; uint8_t v_isShared_87_; uint8_t v_isSharedCheck_100_; 
v_before_84_ = lean_ctor_get(v_head_79_, 0);
v_isSharedCheck_100_ = !lean_is_exclusive(v_head_79_);
if (v_isSharedCheck_100_ == 0)
{
lean_object* v_unused_101_; 
v_unused_101_ = lean_ctor_get(v_head_79_, 1);
lean_dec(v_unused_101_);
v___x_86_ = v_head_79_;
v_isShared_87_ = v_isSharedCheck_100_;
goto v_resetjp_85_;
}
else
{
lean_inc(v_before_84_);
lean_dec(v_head_79_);
v___x_86_ = lean_box(0);
v_isShared_87_ = v_isSharedCheck_100_;
goto v_resetjp_85_;
}
v_resetjp_85_:
{
lean_object* v___x_88_; lean_object* v___x_90_; 
v___x_88_ = lean_obj_once(&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_simpIntroCore_spec__1_spec__1_spec__3_spec__5___closed__0, &lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_simpIntroCore_spec__1_spec__1_spec__3_spec__5___closed__0_once, _init_lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_simpIntroCore_spec__1_spec__1_spec__3_spec__5___closed__0);
if (v_isShared_87_ == 0)
{
lean_ctor_set_tag(v___x_86_, 7);
lean_ctor_set(v___x_86_, 1, v___x_88_);
lean_ctor_set(v___x_86_, 0, v_x_77_);
v___x_90_ = v___x_86_;
goto v_reusejp_89_;
}
else
{
lean_object* v_reuseFailAlloc_99_; 
v_reuseFailAlloc_99_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_99_, 0, v_x_77_);
lean_ctor_set(v_reuseFailAlloc_99_, 1, v___x_88_);
v___x_90_ = v_reuseFailAlloc_99_;
goto v_reusejp_89_;
}
v_reusejp_89_:
{
lean_object* v___x_91_; lean_object* v___x_93_; 
v___x_91_ = lean_obj_once(&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_simpIntroCore_spec__1_spec__1_spec__3_spec__5___closed__3, &lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_simpIntroCore_spec__1_spec__1_spec__3_spec__5___closed__3_once, _init_lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_simpIntroCore_spec__1_spec__1_spec__3_spec__5___closed__3);
if (v_isShared_83_ == 0)
{
lean_ctor_set_tag(v___x_82_, 7);
lean_ctor_set(v___x_82_, 1, v___x_91_);
lean_ctor_set(v___x_82_, 0, v___x_90_);
v___x_93_ = v___x_82_;
goto v_reusejp_92_;
}
else
{
lean_object* v_reuseFailAlloc_98_; 
v_reuseFailAlloc_98_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_98_, 0, v___x_90_);
lean_ctor_set(v_reuseFailAlloc_98_, 1, v___x_91_);
v___x_93_ = v_reuseFailAlloc_98_;
goto v_reusejp_92_;
}
v_reusejp_92_:
{
lean_object* v___x_94_; lean_object* v___x_95_; lean_object* v___x_96_; 
v___x_94_ = l_Lean_MessageData_ofSyntax(v_before_84_);
v___x_95_ = l_Lean_indentD(v___x_94_);
v___x_96_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_96_, 0, v___x_93_);
lean_ctor_set(v___x_96_, 1, v___x_95_);
v_x_77_ = v___x_96_;
v_x_78_ = v_tail_80_;
goto _start;
}
}
}
}
}
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_simpIntroCore_spec__1_spec__1_spec__3_spec__4(lean_object* v_opts_103_, lean_object* v_opt_104_){
_start:
{
lean_object* v_name_105_; lean_object* v_defValue_106_; lean_object* v_map_107_; lean_object* v___x_108_; 
v_name_105_ = lean_ctor_get(v_opt_104_, 0);
v_defValue_106_ = lean_ctor_get(v_opt_104_, 1);
v_map_107_ = lean_ctor_get(v_opts_103_, 0);
v___x_108_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_107_, v_name_105_);
if (lean_obj_tag(v___x_108_) == 0)
{
uint8_t v___x_109_; 
v___x_109_ = lean_unbox(v_defValue_106_);
return v___x_109_;
}
else
{
lean_object* v_val_110_; 
v_val_110_ = lean_ctor_get(v___x_108_, 0);
lean_inc(v_val_110_);
lean_dec_ref_known(v___x_108_, 1);
if (lean_obj_tag(v_val_110_) == 1)
{
uint8_t v_v_111_; 
v_v_111_ = lean_ctor_get_uint8(v_val_110_, 0);
lean_dec_ref_known(v_val_110_, 0);
return v_v_111_;
}
else
{
uint8_t v___x_112_; 
lean_dec(v_val_110_);
v___x_112_ = lean_unbox(v_defValue_106_);
return v___x_112_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_simpIntroCore_spec__1_spec__1_spec__3_spec__4___boxed(lean_object* v_opts_113_, lean_object* v_opt_114_){
_start:
{
uint8_t v_res_115_; lean_object* v_r_116_; 
v_res_115_ = lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_simpIntroCore_spec__1_spec__1_spec__3_spec__4(v_opts_113_, v_opt_114_);
lean_dec_ref(v_opt_114_);
lean_dec_ref(v_opts_113_);
v_r_116_ = lean_box(v_res_115_);
return v_r_116_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_simpIntroCore_spec__1_spec__1_spec__3___redArg___closed__2(void){
_start:
{
lean_object* v___x_120_; lean_object* v___x_121_; 
v___x_120_ = ((lean_object*)(lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_simpIntroCore_spec__1_spec__1_spec__3___redArg___closed__1));
v___x_121_ = l_Lean_MessageData_ofFormat(v___x_120_);
return v___x_121_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_simpIntroCore_spec__1_spec__1_spec__3___redArg(lean_object* v_msgData_122_, lean_object* v_macroStack_123_, lean_object* v___y_124_){
_start:
{
lean_object* v_options_126_; lean_object* v___x_127_; uint8_t v___x_128_; 
v_options_126_ = lean_ctor_get(v___y_124_, 2);
v___x_127_ = l_Lean_Elab_pp_macroStack;
v___x_128_ = lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_simpIntroCore_spec__1_spec__1_spec__3_spec__4(v_options_126_, v___x_127_);
if (v___x_128_ == 0)
{
lean_object* v___x_129_; 
lean_dec(v_macroStack_123_);
v___x_129_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_129_, 0, v_msgData_122_);
return v___x_129_;
}
else
{
if (lean_obj_tag(v_macroStack_123_) == 0)
{
lean_object* v___x_130_; 
v___x_130_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_130_, 0, v_msgData_122_);
return v___x_130_;
}
else
{
lean_object* v_head_131_; lean_object* v_after_132_; lean_object* v___x_134_; uint8_t v_isShared_135_; uint8_t v_isSharedCheck_147_; 
v_head_131_ = lean_ctor_get(v_macroStack_123_, 0);
lean_inc(v_head_131_);
v_after_132_ = lean_ctor_get(v_head_131_, 1);
v_isSharedCheck_147_ = !lean_is_exclusive(v_head_131_);
if (v_isSharedCheck_147_ == 0)
{
lean_object* v_unused_148_; 
v_unused_148_ = lean_ctor_get(v_head_131_, 0);
lean_dec(v_unused_148_);
v___x_134_ = v_head_131_;
v_isShared_135_ = v_isSharedCheck_147_;
goto v_resetjp_133_;
}
else
{
lean_inc(v_after_132_);
lean_dec(v_head_131_);
v___x_134_ = lean_box(0);
v_isShared_135_ = v_isSharedCheck_147_;
goto v_resetjp_133_;
}
v_resetjp_133_:
{
lean_object* v___x_136_; lean_object* v___x_138_; 
v___x_136_ = lean_obj_once(&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_simpIntroCore_spec__1_spec__1_spec__3_spec__5___closed__0, &lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_simpIntroCore_spec__1_spec__1_spec__3_spec__5___closed__0_once, _init_lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_simpIntroCore_spec__1_spec__1_spec__3_spec__5___closed__0);
if (v_isShared_135_ == 0)
{
lean_ctor_set_tag(v___x_134_, 7);
lean_ctor_set(v___x_134_, 1, v___x_136_);
lean_ctor_set(v___x_134_, 0, v_msgData_122_);
v___x_138_ = v___x_134_;
goto v_reusejp_137_;
}
else
{
lean_object* v_reuseFailAlloc_146_; 
v_reuseFailAlloc_146_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_146_, 0, v_msgData_122_);
lean_ctor_set(v_reuseFailAlloc_146_, 1, v___x_136_);
v___x_138_ = v_reuseFailAlloc_146_;
goto v_reusejp_137_;
}
v_reusejp_137_:
{
lean_object* v___x_139_; lean_object* v___x_140_; lean_object* v___x_141_; lean_object* v___x_142_; lean_object* v_msgData_143_; lean_object* v___x_144_; lean_object* v___x_145_; 
v___x_139_ = lean_obj_once(&lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_simpIntroCore_spec__1_spec__1_spec__3___redArg___closed__2, &lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_simpIntroCore_spec__1_spec__1_spec__3___redArg___closed__2_once, _init_lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_simpIntroCore_spec__1_spec__1_spec__3___redArg___closed__2);
v___x_140_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_140_, 0, v___x_138_);
lean_ctor_set(v___x_140_, 1, v___x_139_);
v___x_141_ = l_Lean_MessageData_ofSyntax(v_after_132_);
v___x_142_ = l_Lean_indentD(v___x_141_);
v_msgData_143_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_msgData_143_, 0, v___x_140_);
lean_ctor_set(v_msgData_143_, 1, v___x_142_);
v___x_144_ = lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_simpIntroCore_spec__1_spec__1_spec__3_spec__5(v_msgData_143_, v_macroStack_123_);
v___x_145_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_145_, 0, v___x_144_);
return v___x_145_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_simpIntroCore_spec__1_spec__1_spec__3___redArg___boxed(lean_object* v_msgData_149_, lean_object* v_macroStack_150_, lean_object* v___y_151_, lean_object* v___y_152_){
_start:
{
lean_object* v_res_153_; 
v_res_153_ = lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_simpIntroCore_spec__1_spec__1_spec__3___redArg(v_msgData_149_, v_macroStack_150_, v___y_151_);
lean_dec_ref(v___y_151_);
return v_res_153_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_simpIntroCore_spec__1_spec__1_spec__2(lean_object* v_msgData_154_, lean_object* v___y_155_, lean_object* v___y_156_, lean_object* v___y_157_, lean_object* v___y_158_){
_start:
{
lean_object* v___x_160_; lean_object* v_env_161_; lean_object* v___x_162_; lean_object* v_mctx_163_; lean_object* v_lctx_164_; lean_object* v_options_165_; lean_object* v___x_166_; lean_object* v___x_167_; lean_object* v___x_168_; 
v___x_160_ = lean_st_ref_get(v___y_158_);
v_env_161_ = lean_ctor_get(v___x_160_, 0);
lean_inc_ref(v_env_161_);
lean_dec(v___x_160_);
v___x_162_ = lean_st_ref_get(v___y_156_);
v_mctx_163_ = lean_ctor_get(v___x_162_, 0);
lean_inc_ref(v_mctx_163_);
lean_dec(v___x_162_);
v_lctx_164_ = lean_ctor_get(v___y_155_, 2);
v_options_165_ = lean_ctor_get(v___y_157_, 2);
lean_inc_ref(v_options_165_);
lean_inc_ref(v_lctx_164_);
v___x_166_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_166_, 0, v_env_161_);
lean_ctor_set(v___x_166_, 1, v_mctx_163_);
lean_ctor_set(v___x_166_, 2, v_lctx_164_);
lean_ctor_set(v___x_166_, 3, v_options_165_);
v___x_167_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_167_, 0, v___x_166_);
lean_ctor_set(v___x_167_, 1, v_msgData_154_);
v___x_168_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_168_, 0, v___x_167_);
return v___x_168_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_simpIntroCore_spec__1_spec__1_spec__2___boxed(lean_object* v_msgData_169_, lean_object* v___y_170_, lean_object* v___y_171_, lean_object* v___y_172_, lean_object* v___y_173_, lean_object* v___y_174_){
_start:
{
lean_object* v_res_175_; 
v_res_175_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_simpIntroCore_spec__1_spec__1_spec__2(v_msgData_169_, v___y_170_, v___y_171_, v___y_172_, v___y_173_);
lean_dec(v___y_173_);
lean_dec_ref(v___y_172_);
lean_dec(v___y_171_);
lean_dec_ref(v___y_170_);
return v_res_175_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_simpIntroCore_spec__1_spec__1___redArg(lean_object* v_msg_176_, lean_object* v___y_177_, lean_object* v___y_178_, lean_object* v___y_179_, lean_object* v___y_180_, lean_object* v___y_181_, lean_object* v___y_182_){
_start:
{
lean_object* v_ref_184_; lean_object* v___x_185_; lean_object* v_a_186_; lean_object* v_macroStack_187_; lean_object* v___x_188_; lean_object* v___x_189_; lean_object* v_a_190_; lean_object* v___x_192_; uint8_t v_isShared_193_; uint8_t v_isSharedCheck_198_; 
v_ref_184_ = lean_ctor_get(v___y_181_, 5);
v___x_185_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_simpIntroCore_spec__1_spec__1_spec__2(v_msg_176_, v___y_179_, v___y_180_, v___y_181_, v___y_182_);
v_a_186_ = lean_ctor_get(v___x_185_, 0);
lean_inc(v_a_186_);
lean_dec_ref(v___x_185_);
v_macroStack_187_ = lean_ctor_get(v___y_177_, 1);
v___x_188_ = l_Lean_Elab_getBetterRef(v_ref_184_, v_macroStack_187_);
lean_inc(v_macroStack_187_);
v___x_189_ = lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_simpIntroCore_spec__1_spec__1_spec__3___redArg(v_a_186_, v_macroStack_187_, v___y_181_);
v_a_190_ = lean_ctor_get(v___x_189_, 0);
v_isSharedCheck_198_ = !lean_is_exclusive(v___x_189_);
if (v_isSharedCheck_198_ == 0)
{
v___x_192_ = v___x_189_;
v_isShared_193_ = v_isSharedCheck_198_;
goto v_resetjp_191_;
}
else
{
lean_inc(v_a_190_);
lean_dec(v___x_189_);
v___x_192_ = lean_box(0);
v_isShared_193_ = v_isSharedCheck_198_;
goto v_resetjp_191_;
}
v_resetjp_191_:
{
lean_object* v___x_194_; lean_object* v___x_196_; 
v___x_194_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_194_, 0, v___x_188_);
lean_ctor_set(v___x_194_, 1, v_a_190_);
if (v_isShared_193_ == 0)
{
lean_ctor_set_tag(v___x_192_, 1);
lean_ctor_set(v___x_192_, 0, v___x_194_);
v___x_196_ = v___x_192_;
goto v_reusejp_195_;
}
else
{
lean_object* v_reuseFailAlloc_197_; 
v_reuseFailAlloc_197_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_197_, 0, v___x_194_);
v___x_196_ = v_reuseFailAlloc_197_;
goto v_reusejp_195_;
}
v_reusejp_195_:
{
return v___x_196_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_simpIntroCore_spec__1_spec__1___redArg___boxed(lean_object* v_msg_199_, lean_object* v___y_200_, lean_object* v___y_201_, lean_object* v___y_202_, lean_object* v___y_203_, lean_object* v___y_204_, lean_object* v___y_205_, lean_object* v___y_206_){
_start:
{
lean_object* v_res_207_; 
v_res_207_ = lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_simpIntroCore_spec__1_spec__1___redArg(v_msg_199_, v___y_200_, v___y_201_, v___y_202_, v___y_203_, v___y_204_, v___y_205_);
lean_dec(v___y_205_);
lean_dec_ref(v___y_204_);
lean_dec(v___y_203_);
lean_dec_ref(v___y_202_);
lean_dec(v___y_201_);
lean_dec_ref(v___y_200_);
return v_res_207_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Mathlib_Tactic_simpIntroCore_spec__1___redArg(lean_object* v_ref_208_, lean_object* v_msg_209_, lean_object* v___y_210_, lean_object* v___y_211_, lean_object* v___y_212_, lean_object* v___y_213_, lean_object* v___y_214_, lean_object* v___y_215_){
_start:
{
lean_object* v_fileName_217_; lean_object* v_fileMap_218_; lean_object* v_options_219_; lean_object* v_currRecDepth_220_; lean_object* v_maxRecDepth_221_; lean_object* v_ref_222_; lean_object* v_currNamespace_223_; lean_object* v_openDecls_224_; lean_object* v_initHeartbeats_225_; lean_object* v_maxHeartbeats_226_; lean_object* v_quotContext_227_; lean_object* v_currMacroScope_228_; uint8_t v_diag_229_; lean_object* v_cancelTk_x3f_230_; uint8_t v_suppressElabErrors_231_; lean_object* v_inheritedTraceOptions_232_; lean_object* v_ref_233_; lean_object* v___x_234_; lean_object* v___x_235_; 
v_fileName_217_ = lean_ctor_get(v___y_214_, 0);
v_fileMap_218_ = lean_ctor_get(v___y_214_, 1);
v_options_219_ = lean_ctor_get(v___y_214_, 2);
v_currRecDepth_220_ = lean_ctor_get(v___y_214_, 3);
v_maxRecDepth_221_ = lean_ctor_get(v___y_214_, 4);
v_ref_222_ = lean_ctor_get(v___y_214_, 5);
v_currNamespace_223_ = lean_ctor_get(v___y_214_, 6);
v_openDecls_224_ = lean_ctor_get(v___y_214_, 7);
v_initHeartbeats_225_ = lean_ctor_get(v___y_214_, 8);
v_maxHeartbeats_226_ = lean_ctor_get(v___y_214_, 9);
v_quotContext_227_ = lean_ctor_get(v___y_214_, 10);
v_currMacroScope_228_ = lean_ctor_get(v___y_214_, 11);
v_diag_229_ = lean_ctor_get_uint8(v___y_214_, sizeof(void*)*14);
v_cancelTk_x3f_230_ = lean_ctor_get(v___y_214_, 12);
v_suppressElabErrors_231_ = lean_ctor_get_uint8(v___y_214_, sizeof(void*)*14 + 1);
v_inheritedTraceOptions_232_ = lean_ctor_get(v___y_214_, 13);
v_ref_233_ = l_Lean_replaceRef(v_ref_208_, v_ref_222_);
lean_inc_ref(v_inheritedTraceOptions_232_);
lean_inc(v_cancelTk_x3f_230_);
lean_inc(v_currMacroScope_228_);
lean_inc(v_quotContext_227_);
lean_inc(v_maxHeartbeats_226_);
lean_inc(v_initHeartbeats_225_);
lean_inc(v_openDecls_224_);
lean_inc(v_currNamespace_223_);
lean_inc(v_maxRecDepth_221_);
lean_inc(v_currRecDepth_220_);
lean_inc_ref(v_options_219_);
lean_inc_ref(v_fileMap_218_);
lean_inc_ref(v_fileName_217_);
v___x_234_ = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(v___x_234_, 0, v_fileName_217_);
lean_ctor_set(v___x_234_, 1, v_fileMap_218_);
lean_ctor_set(v___x_234_, 2, v_options_219_);
lean_ctor_set(v___x_234_, 3, v_currRecDepth_220_);
lean_ctor_set(v___x_234_, 4, v_maxRecDepth_221_);
lean_ctor_set(v___x_234_, 5, v_ref_233_);
lean_ctor_set(v___x_234_, 6, v_currNamespace_223_);
lean_ctor_set(v___x_234_, 7, v_openDecls_224_);
lean_ctor_set(v___x_234_, 8, v_initHeartbeats_225_);
lean_ctor_set(v___x_234_, 9, v_maxHeartbeats_226_);
lean_ctor_set(v___x_234_, 10, v_quotContext_227_);
lean_ctor_set(v___x_234_, 11, v_currMacroScope_228_);
lean_ctor_set(v___x_234_, 12, v_cancelTk_x3f_230_);
lean_ctor_set(v___x_234_, 13, v_inheritedTraceOptions_232_);
lean_ctor_set_uint8(v___x_234_, sizeof(void*)*14, v_diag_229_);
lean_ctor_set_uint8(v___x_234_, sizeof(void*)*14 + 1, v_suppressElabErrors_231_);
v___x_235_ = lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_simpIntroCore_spec__1_spec__1___redArg(v_msg_209_, v___y_210_, v___y_211_, v___y_212_, v___y_213_, v___x_234_, v___y_215_);
lean_dec_ref_known(v___x_234_, 14);
return v___x_235_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Mathlib_Tactic_simpIntroCore_spec__1___redArg___boxed(lean_object* v_ref_236_, lean_object* v_msg_237_, lean_object* v___y_238_, lean_object* v___y_239_, lean_object* v___y_240_, lean_object* v___y_241_, lean_object* v___y_242_, lean_object* v___y_243_, lean_object* v___y_244_){
_start:
{
lean_object* v_res_245_; 
v_res_245_ = lp_mathlib_Lean_throwErrorAt___at___00Mathlib_Tactic_simpIntroCore_spec__1___redArg(v_ref_236_, v_msg_237_, v___y_238_, v___y_239_, v___y_240_, v___y_241_, v___y_242_, v___y_243_);
lean_dec(v___y_243_);
lean_dec_ref(v___y_242_);
lean_dec(v___y_241_);
lean_dec_ref(v___y_240_);
lean_dec(v___y_239_);
lean_dec_ref(v___y_238_);
lean_dec(v_ref_236_);
return v_res_245_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_simpIntroCore___lam__0___boxed(lean_object* v_fst_246_, lean_object* v___x_247_, lean_object* v_fst_248_, lean_object* v_snd_249_, lean_object* v_ctx_250_, lean_object* v_simprocs_251_, lean_object* v_discharge_x3f_252_, lean_object* v_more_253_, lean_object* v_snd_254_, lean_object* v___y_255_, lean_object* v___y_256_, lean_object* v___y_257_, lean_object* v___y_258_, lean_object* v___y_259_, lean_object* v___y_260_, lean_object* v___y_261_){
_start:
{
uint8_t v_more_boxed_262_; lean_object* v_res_263_; 
v_more_boxed_262_ = lean_unbox(v_more_253_);
v_res_263_ = lp_mathlib_Mathlib_Tactic_simpIntroCore___lam__0(v_fst_246_, v___x_247_, v_fst_248_, v_snd_249_, v_ctx_250_, v_simprocs_251_, v_discharge_x3f_252_, v_more_boxed_262_, v_snd_254_, v___y_255_, v___y_256_, v___y_257_, v___y_258_, v___y_259_, v___y_260_);
lean_dec(v___y_260_);
lean_dec_ref(v___y_259_);
lean_dec(v___y_258_);
lean_dec_ref(v___y_257_);
lean_dec(v___y_256_);
lean_dec_ref(v___y_255_);
return v_res_263_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_simpIntroCore___closed__1(void){
_start:
{
lean_object* v___x_265_; lean_object* v___x_266_; 
v___x_265_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_simpIntroCore___closed__0));
v___x_266_ = l_Lean_stringToMessageData(v___x_265_);
return v___x_266_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_simpIntroCore___closed__3(void){
_start:
{
lean_object* v___x_268_; lean_object* v___x_269_; 
v___x_268_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_simpIntroCore___closed__2));
v___x_269_ = l_Lean_stringToMessageData(v___x_268_);
return v___x_269_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_simpIntroCore___closed__7(void){
_start:
{
lean_object* v___x_270_; lean_object* v___x_271_; lean_object* v___x_272_; 
v___x_270_ = lean_unsigned_to_nat(32u);
v___x_271_ = lean_mk_empty_array_with_capacity(v___x_270_);
v___x_272_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_272_, 0, v___x_271_);
return v___x_272_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_simpIntroCore___closed__8(void){
_start:
{
size_t v___x_273_; lean_object* v___x_274_; lean_object* v___x_275_; lean_object* v___x_276_; lean_object* v___x_277_; lean_object* v___x_278_; 
v___x_273_ = ((size_t)5ULL);
v___x_274_ = lean_unsigned_to_nat(0u);
v___x_275_ = lean_unsigned_to_nat(32u);
v___x_276_ = lean_mk_empty_array_with_capacity(v___x_275_);
v___x_277_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_simpIntroCore___closed__7, &lp_mathlib_Mathlib_Tactic_simpIntroCore___closed__7_once, _init_lp_mathlib_Mathlib_Tactic_simpIntroCore___closed__7);
v___x_278_ = lean_alloc_ctor(0, 4, sizeof(size_t)*1);
lean_ctor_set(v___x_278_, 0, v___x_277_);
lean_ctor_set(v___x_278_, 1, v___x_276_);
lean_ctor_set(v___x_278_, 2, v___x_274_);
lean_ctor_set(v___x_278_, 3, v___x_274_);
lean_ctor_set_usize(v___x_278_, 4, v___x_273_);
return v___x_278_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_simpIntroCore___closed__4(void){
_start:
{
lean_object* v___x_279_; 
v___x_279_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_279_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_simpIntroCore___closed__5(void){
_start:
{
lean_object* v___x_280_; lean_object* v___x_281_; 
v___x_280_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_simpIntroCore___closed__4, &lp_mathlib_Mathlib_Tactic_simpIntroCore___closed__4_once, _init_lp_mathlib_Mathlib_Tactic_simpIntroCore___closed__4);
v___x_281_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_281_, 0, v___x_280_);
return v___x_281_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_simpIntroCore___closed__9(void){
_start:
{
lean_object* v___x_282_; lean_object* v___x_283_; lean_object* v___x_284_; 
v___x_282_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_simpIntroCore___closed__8, &lp_mathlib_Mathlib_Tactic_simpIntroCore___closed__8_once, _init_lp_mathlib_Mathlib_Tactic_simpIntroCore___closed__8);
v___x_283_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_simpIntroCore___closed__5, &lp_mathlib_Mathlib_Tactic_simpIntroCore___closed__5_once, _init_lp_mathlib_Mathlib_Tactic_simpIntroCore___closed__5);
v___x_284_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_284_, 0, v___x_283_);
lean_ctor_set(v___x_284_, 1, v___x_283_);
lean_ctor_set(v___x_284_, 2, v___x_283_);
lean_ctor_set(v___x_284_, 3, v___x_282_);
return v___x_284_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_simpIntroCore___closed__6(void){
_start:
{
lean_object* v___x_285_; lean_object* v___x_286_; lean_object* v___x_287_; 
v___x_285_ = lean_unsigned_to_nat(0u);
v___x_286_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_simpIntroCore___closed__5, &lp_mathlib_Mathlib_Tactic_simpIntroCore___closed__5_once, _init_lp_mathlib_Mathlib_Tactic_simpIntroCore___closed__5);
v___x_287_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_287_, 0, v___x_286_);
lean_ctor_set(v___x_287_, 1, v___x_285_);
return v___x_287_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_simpIntroCore___closed__10(void){
_start:
{
lean_object* v___x_288_; lean_object* v___x_289_; lean_object* v___x_290_; 
v___x_288_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_simpIntroCore___closed__9, &lp_mathlib_Mathlib_Tactic_simpIntroCore___closed__9_once, _init_lp_mathlib_Mathlib_Tactic_simpIntroCore___closed__9);
v___x_289_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_simpIntroCore___closed__6, &lp_mathlib_Mathlib_Tactic_simpIntroCore___closed__6_once, _init_lp_mathlib_Mathlib_Tactic_simpIntroCore___closed__6);
v___x_290_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_290_, 0, v___x_289_);
lean_ctor_set(v___x_290_, 1, v___x_288_);
return v___x_290_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_simpIntroCore(lean_object* v_g_294_, lean_object* v_ctx_295_, lean_object* v_simprocs_296_, lean_object* v_discharge_x3f_297_, uint8_t v_more_298_, lean_object* v_ids_299_, lean_object* v_a_300_, lean_object* v_a_301_, lean_object* v_a_302_, lean_object* v_a_303_, lean_object* v_a_304_, lean_object* v_a_305_){
_start:
{
lean_object* v___y_308_; lean_object* v___y_309_; lean_object* v_fst_310_; lean_object* v_snd_311_; lean_object* v___y_312_; lean_object* v___y_313_; lean_object* v___y_314_; lean_object* v___y_315_; lean_object* v___y_316_; lean_object* v___y_317_; lean_object* v___y_323_; lean_object* v___y_324_; lean_object* v_x_325_; lean_object* v___y_326_; lean_object* v___y_327_; lean_object* v___y_328_; lean_object* v___y_329_; lean_object* v___y_330_; lean_object* v___y_331_; lean_object* v___y_335_; lean_object* v___y_336_; lean_object* v___y_337_; lean_object* v___y_338_; lean_object* v___y_339_; lean_object* v___y_340_; lean_object* v___y_341_; uint8_t v___x_350_; lean_object* v___x_351_; lean_object* v___x_352_; lean_object* v___y_354_; lean_object* v___y_355_; lean_object* v___y_356_; lean_object* v___y_357_; lean_object* v___y_358_; lean_object* v___y_359_; lean_object* v___y_360_; lean_object* v___y_361_; lean_object* v___y_362_; lean_object* v___y_363_; lean_object* v___y_364_; lean_object* v___y_441_; lean_object* v___y_442_; lean_object* v___y_443_; lean_object* v___y_444_; lean_object* v___y_445_; lean_object* v___y_446_; lean_object* v___y_447_; lean_object* v___y_448_; lean_object* v___y_449_; lean_object* v_a_450_; uint8_t v_fst_455_; lean_object* v_fst_456_; lean_object* v_snd_457_; lean_object* v___y_458_; lean_object* v___y_459_; lean_object* v___y_460_; lean_object* v___y_461_; lean_object* v___y_462_; lean_object* v___y_463_; 
v___x_350_ = 1;
v___x_351_ = lean_unsigned_to_nat(0u);
v___x_352_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_simpIntroCore___closed__10, &lp_mathlib_Mathlib_Tactic_simpIntroCore___closed__10_once, _init_lp_mathlib_Mathlib_Tactic_simpIntroCore___closed__10);
if (lean_obj_tag(v_ids_299_) == 0)
{
if (v_more_298_ == 0)
{
lean_object* v___x_488_; 
v___x_488_ = l_Lean_Meta_simpTargetCore(v_g_294_, v_ctx_295_, v_simprocs_296_, v_discharge_x3f_297_, v___x_350_, v___x_352_, v_a_302_, v_a_303_, v_a_304_, v_a_305_);
if (lean_obj_tag(v___x_488_) == 0)
{
lean_object* v_a_489_; lean_object* v___x_491_; uint8_t v_isShared_492_; uint8_t v_isSharedCheck_497_; 
v_a_489_ = lean_ctor_get(v___x_488_, 0);
v_isSharedCheck_497_ = !lean_is_exclusive(v___x_488_);
if (v_isSharedCheck_497_ == 0)
{
v___x_491_ = v___x_488_;
v_isShared_492_ = v_isSharedCheck_497_;
goto v_resetjp_490_;
}
else
{
lean_inc(v_a_489_);
lean_dec(v___x_488_);
v___x_491_ = lean_box(0);
v_isShared_492_ = v_isSharedCheck_497_;
goto v_resetjp_490_;
}
v_resetjp_490_:
{
lean_object* v_fst_493_; lean_object* v___x_495_; 
v_fst_493_ = lean_ctor_get(v_a_489_, 0);
lean_inc(v_fst_493_);
lean_dec(v_a_489_);
if (v_isShared_492_ == 0)
{
lean_ctor_set(v___x_491_, 0, v_fst_493_);
v___x_495_ = v___x_491_;
goto v_reusejp_494_;
}
else
{
lean_object* v_reuseFailAlloc_496_; 
v_reuseFailAlloc_496_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_496_, 0, v_fst_493_);
v___x_495_ = v_reuseFailAlloc_496_;
goto v_reusejp_494_;
}
v_reusejp_494_:
{
return v___x_495_;
}
}
}
else
{
lean_object* v_a_498_; lean_object* v___x_500_; uint8_t v_isShared_501_; uint8_t v_isSharedCheck_505_; 
v_a_498_ = lean_ctor_get(v___x_488_, 0);
v_isSharedCheck_505_ = !lean_is_exclusive(v___x_488_);
if (v_isSharedCheck_505_ == 0)
{
v___x_500_ = v___x_488_;
v_isShared_501_ = v_isSharedCheck_505_;
goto v_resetjp_499_;
}
else
{
lean_inc(v_a_498_);
lean_dec(v___x_488_);
v___x_500_ = lean_box(0);
v_isShared_501_ = v_isSharedCheck_505_;
goto v_resetjp_499_;
}
v_resetjp_499_:
{
lean_object* v___x_503_; 
if (v_isShared_501_ == 0)
{
v___x_503_ = v___x_500_;
goto v_reusejp_502_;
}
else
{
lean_object* v_reuseFailAlloc_504_; 
v_reuseFailAlloc_504_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_504_, 0, v_a_498_);
v___x_503_ = v_reuseFailAlloc_504_;
goto v_reusejp_502_;
}
v_reusejp_502_:
{
return v___x_503_;
}
}
}
}
else
{
lean_object* v_ref_506_; uint8_t v___x_507_; uint8_t v___x_508_; lean_object* v___x_509_; 
v_ref_506_ = lean_ctor_get(v_a_304_, 5);
v___x_507_ = 2;
v___x_508_ = 0;
v___x_509_ = l_Lean_mkHole(v_ref_506_, v___x_508_);
v_fst_455_ = v___x_507_;
v_fst_456_ = v___x_509_;
v_snd_457_ = v_ids_299_;
v___y_458_ = v_a_300_;
v___y_459_ = v_a_301_;
v___y_460_ = v_a_302_;
v___y_461_ = v_a_303_;
v___y_462_ = v_a_304_;
v___y_463_ = v_a_305_;
goto v___jp_454_;
}
}
else
{
lean_object* v_head_510_; lean_object* v_tail_511_; uint8_t v___x_512_; lean_object* v___x_513_; 
v_head_510_ = lean_ctor_get(v_ids_299_, 0);
v_tail_511_ = lean_ctor_get(v_ids_299_, 1);
v___x_512_ = 1;
v___x_513_ = l_Lean_Syntax_getArg(v_head_510_, v___x_351_);
lean_inc(v_tail_511_);
v_fst_455_ = v___x_512_;
v_fst_456_ = v___x_513_;
v_snd_457_ = v_tail_511_;
v___y_458_ = v_a_300_;
v___y_459_ = v_a_301_;
v___y_460_ = v_a_302_;
v___y_461_ = v_a_303_;
v___y_462_ = v_a_304_;
v___y_463_ = v_a_305_;
goto v___jp_454_;
}
v___jp_307_:
{
lean_object* v___x_318_; lean_object* v___x_319_; lean_object* v___f_320_; lean_object* v___x_321_; 
lean_inc(v_fst_310_);
v___x_318_ = l_Lean_mkFVar(v_fst_310_);
v___x_319_ = lean_box(v_more_298_);
lean_inc(v_snd_311_);
v___f_320_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_simpIntroCore___lam__0___boxed), 16, 9);
lean_closure_set(v___f_320_, 0, v___y_308_);
lean_closure_set(v___f_320_, 1, v___x_318_);
lean_closure_set(v___f_320_, 2, v_fst_310_);
lean_closure_set(v___f_320_, 3, v_snd_311_);
lean_closure_set(v___f_320_, 4, v_ctx_295_);
lean_closure_set(v___f_320_, 5, v_simprocs_296_);
lean_closure_set(v___f_320_, 6, v_discharge_x3f_297_);
lean_closure_set(v___f_320_, 7, v___x_319_);
lean_closure_set(v___f_320_, 8, v___y_309_);
v___x_321_ = lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_simpIntroCore_spec__0___redArg(v_snd_311_, v___f_320_, v___y_312_, v___y_313_, v___y_314_, v___y_315_, v___y_316_, v___y_317_);
return v___x_321_;
}
v___jp_322_:
{
lean_object* v_fst_332_; lean_object* v_snd_333_; 
v_fst_332_ = lean_ctor_get(v_x_325_, 0);
lean_inc(v_fst_332_);
v_snd_333_ = lean_ctor_get(v_x_325_, 1);
lean_inc(v_snd_333_);
lean_dec_ref(v_x_325_);
v___y_308_ = v___y_323_;
v___y_309_ = v___y_324_;
v_fst_310_ = v_fst_332_;
v_snd_311_ = v_snd_333_;
v___y_312_ = v___y_326_;
v___y_313_ = v___y_327_;
v___y_314_ = v___y_328_;
v___y_315_ = v___y_329_;
v___y_316_ = v___y_330_;
v___y_317_ = v___y_331_;
goto v___jp_307_;
}
v___jp_334_:
{
lean_object* v___x_342_; lean_object* v___x_343_; lean_object* v___x_344_; lean_object* v___x_345_; lean_object* v___x_346_; lean_object* v___x_347_; lean_object* v___x_348_; lean_object* v___x_349_; 
v___x_342_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_simpIntroCore___closed__1, &lp_mathlib_Mathlib_Tactic_simpIntroCore___closed__1_once, _init_lp_mathlib_Mathlib_Tactic_simpIntroCore___closed__1);
lean_inc(v___y_336_);
v___x_343_ = l_Lean_MessageData_ofSyntax(v___y_336_);
v___x_344_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_344_, 0, v___x_342_);
lean_ctor_set(v___x_344_, 1, v___x_343_);
v___x_345_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_simpIntroCore___closed__3, &lp_mathlib_Mathlib_Tactic_simpIntroCore___closed__3_once, _init_lp_mathlib_Mathlib_Tactic_simpIntroCore___closed__3);
v___x_346_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_346_, 0, v___x_344_);
lean_ctor_set(v___x_346_, 1, v___x_345_);
v___x_347_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_347_, 0, v_g_294_);
v___x_348_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_348_, 0, v___x_346_);
lean_ctor_set(v___x_348_, 1, v___x_347_);
v___x_349_ = lp_mathlib_Lean_throwErrorAt___at___00Mathlib_Tactic_simpIntroCore_spec__1___redArg(v___y_336_, v___x_348_, v___y_337_, v___y_341_, v___y_335_, v___y_338_, v___y_340_, v___y_339_);
lean_dec(v___y_336_);
return v___x_349_;
}
v___jp_353_:
{
switch(lean_obj_tag(v___y_362_))
{
case 8:
{
lean_object* v___x_365_; 
lean_dec_ref_known(v___y_362_, 4);
lean_dec(v___y_357_);
lean_dec(v_ids_299_);
v___x_365_ = l_Lean_MVarId_intro(v_g_294_, v___y_364_, v___y_356_, v___y_359_, v___y_361_, v___y_360_);
if (lean_obj_tag(v___x_365_) == 0)
{
lean_object* v_a_366_; 
v_a_366_ = lean_ctor_get(v___x_365_, 0);
lean_inc(v_a_366_);
lean_dec_ref_known(v___x_365_, 1);
v___y_323_ = v___y_354_;
v___y_324_ = v___y_355_;
v_x_325_ = v_a_366_;
v___y_326_ = v___y_358_;
v___y_327_ = v___y_363_;
v___y_328_ = v___y_356_;
v___y_329_ = v___y_359_;
v___y_330_ = v___y_361_;
v___y_331_ = v___y_360_;
goto v___jp_322_;
}
else
{
lean_object* v_a_367_; lean_object* v___x_369_; uint8_t v_isShared_370_; uint8_t v_isSharedCheck_374_; 
lean_dec(v___y_355_);
lean_dec(v___y_354_);
lean_dec(v_discharge_x3f_297_);
lean_dec_ref(v_simprocs_296_);
lean_dec_ref(v_ctx_295_);
v_a_367_ = lean_ctor_get(v___x_365_, 0);
v_isSharedCheck_374_ = !lean_is_exclusive(v___x_365_);
if (v_isSharedCheck_374_ == 0)
{
v___x_369_ = v___x_365_;
v_isShared_370_ = v_isSharedCheck_374_;
goto v_resetjp_368_;
}
else
{
lean_inc(v_a_367_);
lean_dec(v___x_365_);
v___x_369_ = lean_box(0);
v_isShared_370_ = v_isSharedCheck_374_;
goto v_resetjp_368_;
}
v_resetjp_368_:
{
lean_object* v___x_372_; 
if (v_isShared_370_ == 0)
{
v___x_372_ = v___x_369_;
goto v_reusejp_371_;
}
else
{
lean_object* v_reuseFailAlloc_373_; 
v_reuseFailAlloc_373_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_373_, 0, v_a_367_);
v___x_372_ = v_reuseFailAlloc_373_;
goto v_reusejp_371_;
}
v_reusejp_371_:
{
return v___x_372_;
}
}
}
}
case 7:
{
lean_object* v_body_375_; lean_object* v___x_376_; 
lean_dec(v_ids_299_);
v_body_375_ = lean_ctor_get(v___y_362_, 2);
lean_inc_ref(v_body_375_);
lean_dec_ref_known(v___y_362_, 3);
v___x_376_ = l_Lean_MVarId_intro(v_g_294_, v___y_364_, v___y_356_, v___y_359_, v___y_361_, v___y_360_);
if (lean_obj_tag(v___x_376_) == 0)
{
lean_object* v_a_377_; lean_object* v_fst_378_; lean_object* v_snd_379_; uint8_t v___x_380_; 
v_a_377_ = lean_ctor_get(v___x_376_, 0);
lean_inc(v_a_377_);
lean_dec_ref_known(v___x_376_, 1);
v_fst_378_ = lean_ctor_get(v_a_377_, 0);
lean_inc(v_fst_378_);
v_snd_379_ = lean_ctor_get(v_a_377_, 1);
lean_inc(v_snd_379_);
lean_dec(v_a_377_);
v___x_380_ = l_Lean_Expr_hasLooseBVars(v_body_375_);
lean_dec_ref(v_body_375_);
if (v___x_380_ == 0)
{
lean_object* v___x_381_; 
lean_inc(v_discharge_x3f_297_);
lean_inc_ref(v_simprocs_296_);
lean_inc_ref(v_ctx_295_);
lean_inc(v_fst_378_);
lean_inc(v_snd_379_);
v___x_381_ = l_Lean_Meta_simpLocalDecl(v_snd_379_, v_fst_378_, v_ctx_295_, v_simprocs_296_, v_discharge_x3f_297_, v___x_350_, v___x_352_, v___y_356_, v___y_359_, v___y_361_, v___y_360_);
if (lean_obj_tag(v___x_381_) == 0)
{
lean_object* v_a_382_; lean_object* v_fst_383_; 
v_a_382_ = lean_ctor_get(v___x_381_, 0);
lean_inc(v_a_382_);
lean_dec_ref_known(v___x_381_, 1);
v_fst_383_ = lean_ctor_get(v_a_382_, 0);
lean_inc(v_fst_383_);
lean_dec(v_a_382_);
if (lean_obj_tag(v_fst_383_) == 0)
{
lean_object* v___x_384_; lean_object* v___x_385_; lean_object* v___x_386_; 
lean_dec(v___y_355_);
lean_dec(v___y_354_);
lean_dec(v_discharge_x3f_297_);
lean_dec_ref(v_simprocs_296_);
lean_dec_ref(v_ctx_295_);
v___x_384_ = l_Lean_mkFVar(v_fst_378_);
v___x_385_ = lean_alloc_closure((void*)(l_Lean_Elab_Term_addLocalVarInfo___boxed), 9, 2);
lean_closure_set(v___x_385_, 0, v___y_357_);
lean_closure_set(v___x_385_, 1, v___x_384_);
v___x_386_ = lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_simpIntroCore_spec__0___redArg(v_snd_379_, v___x_385_, v___y_358_, v___y_363_, v___y_356_, v___y_359_, v___y_361_, v___y_360_);
if (lean_obj_tag(v___x_386_) == 0)
{
lean_object* v___x_388_; uint8_t v_isShared_389_; uint8_t v_isSharedCheck_394_; 
v_isSharedCheck_394_ = !lean_is_exclusive(v___x_386_);
if (v_isSharedCheck_394_ == 0)
{
lean_object* v_unused_395_; 
v_unused_395_ = lean_ctor_get(v___x_386_, 0);
lean_dec(v_unused_395_);
v___x_388_ = v___x_386_;
v_isShared_389_ = v_isSharedCheck_394_;
goto v_resetjp_387_;
}
else
{
lean_dec(v___x_386_);
v___x_388_ = lean_box(0);
v_isShared_389_ = v_isSharedCheck_394_;
goto v_resetjp_387_;
}
v_resetjp_387_:
{
lean_object* v___x_390_; lean_object* v___x_392_; 
v___x_390_ = lean_box(0);
if (v_isShared_389_ == 0)
{
lean_ctor_set(v___x_388_, 0, v___x_390_);
v___x_392_ = v___x_388_;
goto v_reusejp_391_;
}
else
{
lean_object* v_reuseFailAlloc_393_; 
v_reuseFailAlloc_393_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_393_, 0, v___x_390_);
v___x_392_ = v_reuseFailAlloc_393_;
goto v_reusejp_391_;
}
v_reusejp_391_:
{
return v___x_392_;
}
}
}
else
{
lean_object* v_a_396_; lean_object* v___x_398_; uint8_t v_isShared_399_; uint8_t v_isSharedCheck_403_; 
v_a_396_ = lean_ctor_get(v___x_386_, 0);
v_isSharedCheck_403_ = !lean_is_exclusive(v___x_386_);
if (v_isSharedCheck_403_ == 0)
{
v___x_398_ = v___x_386_;
v_isShared_399_ = v_isSharedCheck_403_;
goto v_resetjp_397_;
}
else
{
lean_inc(v_a_396_);
lean_dec(v___x_386_);
v___x_398_ = lean_box(0);
v_isShared_399_ = v_isSharedCheck_403_;
goto v_resetjp_397_;
}
v_resetjp_397_:
{
lean_object* v___x_401_; 
if (v_isShared_399_ == 0)
{
v___x_401_ = v___x_398_;
goto v_reusejp_400_;
}
else
{
lean_object* v_reuseFailAlloc_402_; 
v_reuseFailAlloc_402_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_402_, 0, v_a_396_);
v___x_401_ = v_reuseFailAlloc_402_;
goto v_reusejp_400_;
}
v_reusejp_400_:
{
return v___x_401_;
}
}
}
}
else
{
lean_object* v_val_404_; 
lean_dec(v_snd_379_);
lean_dec(v_fst_378_);
lean_dec(v___y_357_);
v_val_404_ = lean_ctor_get(v_fst_383_, 0);
lean_inc(v_val_404_);
lean_dec_ref_known(v_fst_383_, 1);
v___y_323_ = v___y_354_;
v___y_324_ = v___y_355_;
v_x_325_ = v_val_404_;
v___y_326_ = v___y_358_;
v___y_327_ = v___y_363_;
v___y_328_ = v___y_356_;
v___y_329_ = v___y_359_;
v___y_330_ = v___y_361_;
v___y_331_ = v___y_360_;
goto v___jp_322_;
}
}
else
{
lean_object* v_a_405_; lean_object* v___x_407_; uint8_t v_isShared_408_; uint8_t v_isSharedCheck_412_; 
lean_dec(v_snd_379_);
lean_dec(v_fst_378_);
lean_dec(v___y_357_);
lean_dec(v___y_355_);
lean_dec(v___y_354_);
lean_dec(v_discharge_x3f_297_);
lean_dec_ref(v_simprocs_296_);
lean_dec_ref(v_ctx_295_);
v_a_405_ = lean_ctor_get(v___x_381_, 0);
v_isSharedCheck_412_ = !lean_is_exclusive(v___x_381_);
if (v_isSharedCheck_412_ == 0)
{
v___x_407_ = v___x_381_;
v_isShared_408_ = v_isSharedCheck_412_;
goto v_resetjp_406_;
}
else
{
lean_inc(v_a_405_);
lean_dec(v___x_381_);
v___x_407_ = lean_box(0);
v_isShared_408_ = v_isSharedCheck_412_;
goto v_resetjp_406_;
}
v_resetjp_406_:
{
lean_object* v___x_410_; 
if (v_isShared_408_ == 0)
{
v___x_410_ = v___x_407_;
goto v_reusejp_409_;
}
else
{
lean_object* v_reuseFailAlloc_411_; 
v_reuseFailAlloc_411_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_411_, 0, v_a_405_);
v___x_410_ = v_reuseFailAlloc_411_;
goto v_reusejp_409_;
}
v_reusejp_409_:
{
return v___x_410_;
}
}
}
}
else
{
lean_dec(v___y_357_);
v___y_308_ = v___y_354_;
v___y_309_ = v___y_355_;
v_fst_310_ = v_fst_378_;
v_snd_311_ = v_snd_379_;
v___y_312_ = v___y_358_;
v___y_313_ = v___y_363_;
v___y_314_ = v___y_356_;
v___y_315_ = v___y_359_;
v___y_316_ = v___y_361_;
v___y_317_ = v___y_360_;
goto v___jp_307_;
}
}
else
{
lean_object* v_a_413_; lean_object* v___x_415_; uint8_t v_isShared_416_; uint8_t v_isSharedCheck_420_; 
lean_dec_ref(v_body_375_);
lean_dec(v___y_357_);
lean_dec(v___y_355_);
lean_dec(v___y_354_);
lean_dec(v_discharge_x3f_297_);
lean_dec_ref(v_simprocs_296_);
lean_dec_ref(v_ctx_295_);
v_a_413_ = lean_ctor_get(v___x_376_, 0);
v_isSharedCheck_420_ = !lean_is_exclusive(v___x_376_);
if (v_isSharedCheck_420_ == 0)
{
v___x_415_ = v___x_376_;
v_isShared_416_ = v_isSharedCheck_420_;
goto v_resetjp_414_;
}
else
{
lean_inc(v_a_413_);
lean_dec(v___x_376_);
v___x_415_ = lean_box(0);
v_isShared_416_ = v_isSharedCheck_420_;
goto v_resetjp_414_;
}
v_resetjp_414_:
{
lean_object* v___x_418_; 
if (v_isShared_416_ == 0)
{
v___x_418_ = v___x_415_;
goto v_reusejp_417_;
}
else
{
lean_object* v_reuseFailAlloc_419_; 
v_reuseFailAlloc_419_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_419_, 0, v_a_413_);
v___x_418_ = v_reuseFailAlloc_419_;
goto v_reusejp_417_;
}
v_reusejp_417_:
{
return v___x_418_;
}
}
}
}
default: 
{
lean_dec(v___y_364_);
lean_dec_ref(v___y_362_);
lean_dec(v___y_355_);
lean_dec(v___y_354_);
if (v_more_298_ == 0)
{
lean_dec(v_ids_299_);
lean_dec(v_discharge_x3f_297_);
lean_dec_ref(v_simprocs_296_);
lean_dec_ref(v_ctx_295_);
v___y_335_ = v___y_356_;
v___y_336_ = v___y_357_;
v___y_337_ = v___y_358_;
v___y_338_ = v___y_359_;
v___y_339_ = v___y_360_;
v___y_340_ = v___y_361_;
v___y_341_ = v___y_363_;
goto v___jp_334_;
}
else
{
uint8_t v___x_421_; 
v___x_421_ = l_List_isEmpty___redArg(v_ids_299_);
lean_dec(v_ids_299_);
if (v___x_421_ == 0)
{
lean_dec(v_discharge_x3f_297_);
lean_dec_ref(v_simprocs_296_);
lean_dec_ref(v_ctx_295_);
v___y_335_ = v___y_356_;
v___y_336_ = v___y_357_;
v___y_337_ = v___y_358_;
v___y_338_ = v___y_359_;
v___y_339_ = v___y_360_;
v___y_340_ = v___y_361_;
v___y_341_ = v___y_363_;
goto v___jp_334_;
}
else
{
lean_object* v___x_422_; 
lean_dec(v___y_357_);
v___x_422_ = l_Lean_Meta_simpTargetCore(v_g_294_, v_ctx_295_, v_simprocs_296_, v_discharge_x3f_297_, v___x_350_, v___x_352_, v___y_356_, v___y_359_, v___y_361_, v___y_360_);
if (lean_obj_tag(v___x_422_) == 0)
{
lean_object* v_a_423_; lean_object* v___x_425_; uint8_t v_isShared_426_; uint8_t v_isSharedCheck_431_; 
v_a_423_ = lean_ctor_get(v___x_422_, 0);
v_isSharedCheck_431_ = !lean_is_exclusive(v___x_422_);
if (v_isSharedCheck_431_ == 0)
{
v___x_425_ = v___x_422_;
v_isShared_426_ = v_isSharedCheck_431_;
goto v_resetjp_424_;
}
else
{
lean_inc(v_a_423_);
lean_dec(v___x_422_);
v___x_425_ = lean_box(0);
v_isShared_426_ = v_isSharedCheck_431_;
goto v_resetjp_424_;
}
v_resetjp_424_:
{
lean_object* v_fst_427_; lean_object* v___x_429_; 
v_fst_427_ = lean_ctor_get(v_a_423_, 0);
lean_inc(v_fst_427_);
lean_dec(v_a_423_);
if (v_isShared_426_ == 0)
{
lean_ctor_set(v___x_425_, 0, v_fst_427_);
v___x_429_ = v___x_425_;
goto v_reusejp_428_;
}
else
{
lean_object* v_reuseFailAlloc_430_; 
v_reuseFailAlloc_430_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_430_, 0, v_fst_427_);
v___x_429_ = v_reuseFailAlloc_430_;
goto v_reusejp_428_;
}
v_reusejp_428_:
{
return v___x_429_;
}
}
}
else
{
lean_object* v_a_432_; lean_object* v___x_434_; uint8_t v_isShared_435_; uint8_t v_isSharedCheck_439_; 
v_a_432_ = lean_ctor_get(v___x_422_, 0);
v_isSharedCheck_439_ = !lean_is_exclusive(v___x_422_);
if (v_isSharedCheck_439_ == 0)
{
v___x_434_ = v___x_422_;
v_isShared_435_ = v_isSharedCheck_439_;
goto v_resetjp_433_;
}
else
{
lean_inc(v_a_432_);
lean_dec(v___x_422_);
v___x_434_ = lean_box(0);
v_isShared_435_ = v_isSharedCheck_439_;
goto v_resetjp_433_;
}
v_resetjp_433_:
{
lean_object* v___x_437_; 
if (v_isShared_435_ == 0)
{
v___x_437_ = v___x_434_;
goto v_reusejp_436_;
}
else
{
lean_object* v_reuseFailAlloc_438_; 
v_reuseFailAlloc_438_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_438_, 0, v_a_432_);
v___x_437_ = v_reuseFailAlloc_438_;
goto v_reusejp_436_;
}
v_reusejp_436_:
{
return v___x_437_;
}
}
}
}
}
}
}
}
v___jp_440_:
{
uint8_t v___x_451_; 
v___x_451_ = l_Lean_Syntax_isIdent(v___y_444_);
if (v___x_451_ == 0)
{
lean_object* v___x_452_; 
v___x_452_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_simpIntroCore___closed__12));
v___y_354_ = v___y_441_;
v___y_355_ = v___y_442_;
v___y_356_ = v___y_443_;
v___y_357_ = v___y_444_;
v___y_358_ = v___y_445_;
v___y_359_ = v___y_446_;
v___y_360_ = v___y_448_;
v___y_361_ = v___y_447_;
v___y_362_ = v_a_450_;
v___y_363_ = v___y_449_;
v___y_364_ = v___x_452_;
goto v___jp_353_;
}
else
{
lean_object* v___x_453_; 
v___x_453_ = l_Lean_Syntax_getId(v___y_444_);
v___y_354_ = v___y_441_;
v___y_355_ = v___y_442_;
v___y_356_ = v___y_443_;
v___y_357_ = v___y_444_;
v___y_358_ = v___y_445_;
v___y_359_ = v___y_446_;
v___y_360_ = v___y_448_;
v___y_361_ = v___y_447_;
v___y_362_ = v_a_450_;
v___y_363_ = v___y_449_;
v___y_364_ = v___x_453_;
goto v___jp_353_;
}
}
v___jp_454_:
{
lean_object* v_keyedConfig_464_; uint8_t v_trackZetaDelta_465_; lean_object* v_zetaDeltaSet_466_; lean_object* v_lctx_467_; lean_object* v_localInstances_468_; lean_object* v_defEqCtx_x3f_469_; lean_object* v_synthPendingDepth_470_; lean_object* v_customCanUnfoldPredicate_x3f_471_; uint8_t v_univApprox_472_; uint8_t v_inTypeClassResolution_473_; uint8_t v_cacheInferType_474_; lean_object* v___x_475_; lean_object* v___x_476_; lean_object* v___x_477_; 
v_keyedConfig_464_ = lean_ctor_get(v___y_460_, 0);
v_trackZetaDelta_465_ = lean_ctor_get_uint8(v___y_460_, sizeof(void*)*7);
v_zetaDeltaSet_466_ = lean_ctor_get(v___y_460_, 1);
v_lctx_467_ = lean_ctor_get(v___y_460_, 2);
v_localInstances_468_ = lean_ctor_get(v___y_460_, 3);
v_defEqCtx_x3f_469_ = lean_ctor_get(v___y_460_, 4);
v_synthPendingDepth_470_ = lean_ctor_get(v___y_460_, 5);
v_customCanUnfoldPredicate_x3f_471_ = lean_ctor_get(v___y_460_, 6);
v_univApprox_472_ = lean_ctor_get_uint8(v___y_460_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_473_ = lean_ctor_get_uint8(v___y_460_, sizeof(void*)*7 + 2);
v_cacheInferType_474_ = lean_ctor_get_uint8(v___y_460_, sizeof(void*)*7 + 3);
lean_inc_ref(v_keyedConfig_464_);
v___x_475_ = l_Lean_Meta_ConfigWithKey_setTransparency(v_fst_455_, v_keyedConfig_464_);
lean_inc(v_customCanUnfoldPredicate_x3f_471_);
lean_inc(v_synthPendingDepth_470_);
lean_inc(v_defEqCtx_x3f_469_);
lean_inc_ref(v_localInstances_468_);
lean_inc_ref(v_lctx_467_);
lean_inc(v_zetaDeltaSet_466_);
v___x_476_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v___x_476_, 0, v___x_475_);
lean_ctor_set(v___x_476_, 1, v_zetaDeltaSet_466_);
lean_ctor_set(v___x_476_, 2, v_lctx_467_);
lean_ctor_set(v___x_476_, 3, v_localInstances_468_);
lean_ctor_set(v___x_476_, 4, v_defEqCtx_x3f_469_);
lean_ctor_set(v___x_476_, 5, v_synthPendingDepth_470_);
lean_ctor_set(v___x_476_, 6, v_customCanUnfoldPredicate_x3f_471_);
lean_ctor_set_uint8(v___x_476_, sizeof(void*)*7, v_trackZetaDelta_465_);
lean_ctor_set_uint8(v___x_476_, sizeof(void*)*7 + 1, v_univApprox_472_);
lean_ctor_set_uint8(v___x_476_, sizeof(void*)*7 + 2, v_inTypeClassResolution_473_);
lean_ctor_set_uint8(v___x_476_, sizeof(void*)*7 + 3, v_cacheInferType_474_);
lean_inc(v_g_294_);
v___x_477_ = l_Lean_MVarId_getType_x27(v_g_294_, v___x_476_, v___y_461_, v___y_462_, v___y_463_);
lean_dec_ref_known(v___x_476_, 7);
if (lean_obj_tag(v___x_477_) == 0)
{
lean_object* v_a_478_; 
v_a_478_ = lean_ctor_get(v___x_477_, 0);
lean_inc(v_a_478_);
lean_dec_ref_known(v___x_477_, 1);
lean_inc(v_fst_456_);
v___y_441_ = v_fst_456_;
v___y_442_ = v_snd_457_;
v___y_443_ = v___y_460_;
v___y_444_ = v_fst_456_;
v___y_445_ = v___y_458_;
v___y_446_ = v___y_461_;
v___y_447_ = v___y_462_;
v___y_448_ = v___y_463_;
v___y_449_ = v___y_459_;
v_a_450_ = v_a_478_;
goto v___jp_440_;
}
else
{
if (lean_obj_tag(v___x_477_) == 0)
{
lean_object* v_a_479_; 
v_a_479_ = lean_ctor_get(v___x_477_, 0);
lean_inc(v_a_479_);
lean_dec_ref_known(v___x_477_, 1);
lean_inc(v_fst_456_);
v___y_441_ = v_fst_456_;
v___y_442_ = v_snd_457_;
v___y_443_ = v___y_460_;
v___y_444_ = v_fst_456_;
v___y_445_ = v___y_458_;
v___y_446_ = v___y_461_;
v___y_447_ = v___y_462_;
v___y_448_ = v___y_463_;
v___y_449_ = v___y_459_;
v_a_450_ = v_a_479_;
goto v___jp_440_;
}
else
{
lean_object* v_a_480_; lean_object* v___x_482_; uint8_t v_isShared_483_; uint8_t v_isSharedCheck_487_; 
lean_dec(v_snd_457_);
lean_dec(v_fst_456_);
lean_dec(v_ids_299_);
lean_dec(v_discharge_x3f_297_);
lean_dec_ref(v_simprocs_296_);
lean_dec_ref(v_ctx_295_);
lean_dec(v_g_294_);
v_a_480_ = lean_ctor_get(v___x_477_, 0);
v_isSharedCheck_487_ = !lean_is_exclusive(v___x_477_);
if (v_isSharedCheck_487_ == 0)
{
v___x_482_ = v___x_477_;
v_isShared_483_ = v_isSharedCheck_487_;
goto v_resetjp_481_;
}
else
{
lean_inc(v_a_480_);
lean_dec(v___x_477_);
v___x_482_ = lean_box(0);
v_isShared_483_ = v_isSharedCheck_487_;
goto v_resetjp_481_;
}
v_resetjp_481_:
{
lean_object* v___x_485_; 
if (v_isShared_483_ == 0)
{
v___x_485_ = v___x_482_;
goto v_reusejp_484_;
}
else
{
lean_object* v_reuseFailAlloc_486_; 
v_reuseFailAlloc_486_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_486_, 0, v_a_480_);
v___x_485_ = v_reuseFailAlloc_486_;
goto v_reusejp_484_;
}
v_reusejp_484_:
{
return v___x_485_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_simpIntroCore___lam__0(lean_object* v_fst_514_, lean_object* v___x_515_, lean_object* v_fst_516_, lean_object* v_snd_517_, lean_object* v_ctx_518_, lean_object* v_simprocs_519_, lean_object* v_discharge_x3f_520_, uint8_t v_more_521_, lean_object* v_snd_522_, lean_object* v___y_523_, lean_object* v___y_524_, lean_object* v___y_525_, lean_object* v___y_526_, lean_object* v___y_527_, lean_object* v___y_528_){
_start:
{
lean_object* v___x_530_; 
v___x_530_ = l_Lean_Elab_Term_addLocalVarInfo(v_fst_514_, v___x_515_, v___y_523_, v___y_524_, v___y_525_, v___y_526_, v___y_527_, v___y_528_);
if (lean_obj_tag(v___x_530_) == 0)
{
lean_object* v___x_531_; 
lean_dec_ref_known(v___x_530_, 1);
lean_inc(v_fst_516_);
v___x_531_ = l_Lean_FVarId_getType___redArg(v_fst_516_, v___y_525_, v___y_527_, v___y_528_);
if (lean_obj_tag(v___x_531_) == 0)
{
lean_object* v_a_532_; lean_object* v___x_533_; 
v_a_532_ = lean_ctor_get(v___x_531_, 0);
lean_inc(v_a_532_);
lean_dec_ref_known(v___x_531_, 1);
v___x_533_ = l_Lean_Meta_isProp(v_a_532_, v___y_525_, v___y_526_, v___y_527_, v___y_528_);
if (lean_obj_tag(v___x_533_) == 0)
{
lean_object* v_a_534_; uint8_t v___x_535_; 
v_a_534_ = lean_ctor_get(v___x_533_, 0);
lean_inc(v_a_534_);
lean_dec_ref_known(v___x_533_, 1);
v___x_535_ = lean_unbox(v_a_534_);
lean_dec(v_a_534_);
if (v___x_535_ == 0)
{
lean_object* v___x_536_; 
lean_dec(v_fst_516_);
v___x_536_ = lp_mathlib_Mathlib_Tactic_simpIntroCore(v_snd_517_, v_ctx_518_, v_simprocs_519_, v_discharge_x3f_520_, v_more_521_, v_snd_522_, v___y_523_, v___y_524_, v___y_525_, v___y_526_, v___y_527_, v___y_528_);
return v___x_536_;
}
else
{
lean_object* v_simpTheorems_537_; lean_object* v___x_538_; lean_object* v___x_539_; lean_object* v___x_540_; lean_object* v___x_541_; 
v_simpTheorems_537_ = lean_ctor_get(v_ctx_518_, 6);
lean_inc(v_fst_516_);
v___x_538_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_538_, 0, v_fst_516_);
v___x_539_ = l_Lean_Expr_fvar___override(v_fst_516_);
v___x_540_ = l_Lean_Meta_simpGlobalConfig;
lean_inc_ref(v_simpTheorems_537_);
v___x_541_ = l_Lean_Meta_SimpTheoremsArray_addTheorem(v_simpTheorems_537_, v___x_538_, v___x_539_, v___x_540_, v___y_525_, v___y_526_, v___y_527_, v___y_528_);
if (lean_obj_tag(v___x_541_) == 0)
{
lean_object* v_a_542_; lean_object* v___x_543_; lean_object* v___x_544_; 
v_a_542_ = lean_ctor_get(v___x_541_, 0);
lean_inc(v_a_542_);
lean_dec_ref_known(v___x_541_, 1);
v___x_543_ = l_Lean_Meta_Simp_Context_setSimpTheorems(v_ctx_518_, v_a_542_);
v___x_544_ = lp_mathlib_Mathlib_Tactic_simpIntroCore(v_snd_517_, v___x_543_, v_simprocs_519_, v_discharge_x3f_520_, v_more_521_, v_snd_522_, v___y_523_, v___y_524_, v___y_525_, v___y_526_, v___y_527_, v___y_528_);
return v___x_544_;
}
else
{
lean_object* v_a_545_; lean_object* v___x_547_; uint8_t v_isShared_548_; uint8_t v_isSharedCheck_552_; 
lean_dec(v_snd_522_);
lean_dec(v_discharge_x3f_520_);
lean_dec_ref(v_simprocs_519_);
lean_dec_ref(v_ctx_518_);
lean_dec(v_snd_517_);
v_a_545_ = lean_ctor_get(v___x_541_, 0);
v_isSharedCheck_552_ = !lean_is_exclusive(v___x_541_);
if (v_isSharedCheck_552_ == 0)
{
v___x_547_ = v___x_541_;
v_isShared_548_ = v_isSharedCheck_552_;
goto v_resetjp_546_;
}
else
{
lean_inc(v_a_545_);
lean_dec(v___x_541_);
v___x_547_ = lean_box(0);
v_isShared_548_ = v_isSharedCheck_552_;
goto v_resetjp_546_;
}
v_resetjp_546_:
{
lean_object* v___x_550_; 
if (v_isShared_548_ == 0)
{
v___x_550_ = v___x_547_;
goto v_reusejp_549_;
}
else
{
lean_object* v_reuseFailAlloc_551_; 
v_reuseFailAlloc_551_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_551_, 0, v_a_545_);
v___x_550_ = v_reuseFailAlloc_551_;
goto v_reusejp_549_;
}
v_reusejp_549_:
{
return v___x_550_;
}
}
}
}
}
else
{
lean_object* v_a_553_; lean_object* v___x_555_; uint8_t v_isShared_556_; uint8_t v_isSharedCheck_560_; 
lean_dec(v_snd_522_);
lean_dec(v_discharge_x3f_520_);
lean_dec_ref(v_simprocs_519_);
lean_dec_ref(v_ctx_518_);
lean_dec(v_snd_517_);
lean_dec(v_fst_516_);
v_a_553_ = lean_ctor_get(v___x_533_, 0);
v_isSharedCheck_560_ = !lean_is_exclusive(v___x_533_);
if (v_isSharedCheck_560_ == 0)
{
v___x_555_ = v___x_533_;
v_isShared_556_ = v_isSharedCheck_560_;
goto v_resetjp_554_;
}
else
{
lean_inc(v_a_553_);
lean_dec(v___x_533_);
v___x_555_ = lean_box(0);
v_isShared_556_ = v_isSharedCheck_560_;
goto v_resetjp_554_;
}
v_resetjp_554_:
{
lean_object* v___x_558_; 
if (v_isShared_556_ == 0)
{
v___x_558_ = v___x_555_;
goto v_reusejp_557_;
}
else
{
lean_object* v_reuseFailAlloc_559_; 
v_reuseFailAlloc_559_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_559_, 0, v_a_553_);
v___x_558_ = v_reuseFailAlloc_559_;
goto v_reusejp_557_;
}
v_reusejp_557_:
{
return v___x_558_;
}
}
}
}
else
{
lean_object* v_a_561_; lean_object* v___x_563_; uint8_t v_isShared_564_; uint8_t v_isSharedCheck_568_; 
lean_dec(v_snd_522_);
lean_dec(v_discharge_x3f_520_);
lean_dec_ref(v_simprocs_519_);
lean_dec_ref(v_ctx_518_);
lean_dec(v_snd_517_);
lean_dec(v_fst_516_);
v_a_561_ = lean_ctor_get(v___x_531_, 0);
v_isSharedCheck_568_ = !lean_is_exclusive(v___x_531_);
if (v_isSharedCheck_568_ == 0)
{
v___x_563_ = v___x_531_;
v_isShared_564_ = v_isSharedCheck_568_;
goto v_resetjp_562_;
}
else
{
lean_inc(v_a_561_);
lean_dec(v___x_531_);
v___x_563_ = lean_box(0);
v_isShared_564_ = v_isSharedCheck_568_;
goto v_resetjp_562_;
}
v_resetjp_562_:
{
lean_object* v___x_566_; 
if (v_isShared_564_ == 0)
{
v___x_566_ = v___x_563_;
goto v_reusejp_565_;
}
else
{
lean_object* v_reuseFailAlloc_567_; 
v_reuseFailAlloc_567_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_567_, 0, v_a_561_);
v___x_566_ = v_reuseFailAlloc_567_;
goto v_reusejp_565_;
}
v_reusejp_565_:
{
return v___x_566_;
}
}
}
}
else
{
lean_object* v_a_569_; lean_object* v___x_571_; uint8_t v_isShared_572_; uint8_t v_isSharedCheck_576_; 
lean_dec(v_snd_522_);
lean_dec(v_discharge_x3f_520_);
lean_dec_ref(v_simprocs_519_);
lean_dec_ref(v_ctx_518_);
lean_dec(v_snd_517_);
lean_dec(v_fst_516_);
v_a_569_ = lean_ctor_get(v___x_530_, 0);
v_isSharedCheck_576_ = !lean_is_exclusive(v___x_530_);
if (v_isSharedCheck_576_ == 0)
{
v___x_571_ = v___x_530_;
v_isShared_572_ = v_isSharedCheck_576_;
goto v_resetjp_570_;
}
else
{
lean_inc(v_a_569_);
lean_dec(v___x_530_);
v___x_571_ = lean_box(0);
v_isShared_572_ = v_isSharedCheck_576_;
goto v_resetjp_570_;
}
v_resetjp_570_:
{
lean_object* v___x_574_; 
if (v_isShared_572_ == 0)
{
v___x_574_ = v___x_571_;
goto v_reusejp_573_;
}
else
{
lean_object* v_reuseFailAlloc_575_; 
v_reuseFailAlloc_575_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_575_, 0, v_a_569_);
v___x_574_ = v_reuseFailAlloc_575_;
goto v_reusejp_573_;
}
v_reusejp_573_:
{
return v___x_574_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_simpIntroCore___boxed(lean_object* v_g_577_, lean_object* v_ctx_578_, lean_object* v_simprocs_579_, lean_object* v_discharge_x3f_580_, lean_object* v_more_581_, lean_object* v_ids_582_, lean_object* v_a_583_, lean_object* v_a_584_, lean_object* v_a_585_, lean_object* v_a_586_, lean_object* v_a_587_, lean_object* v_a_588_, lean_object* v_a_589_){
_start:
{
uint8_t v_more_boxed_590_; lean_object* v_res_591_; 
v_more_boxed_590_ = lean_unbox(v_more_581_);
v_res_591_ = lp_mathlib_Mathlib_Tactic_simpIntroCore(v_g_577_, v_ctx_578_, v_simprocs_579_, v_discharge_x3f_580_, v_more_boxed_590_, v_ids_582_, v_a_583_, v_a_584_, v_a_585_, v_a_586_, v_a_587_, v_a_588_);
lean_dec(v_a_588_);
lean_dec_ref(v_a_587_);
lean_dec(v_a_586_);
lean_dec_ref(v_a_585_);
lean_dec(v_a_584_);
lean_dec_ref(v_a_583_);
return v_res_591_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Mathlib_Tactic_simpIntroCore_spec__1(lean_object* v_00_u03b1_592_, lean_object* v_ref_593_, lean_object* v_msg_594_, lean_object* v___y_595_, lean_object* v___y_596_, lean_object* v___y_597_, lean_object* v___y_598_, lean_object* v___y_599_, lean_object* v___y_600_){
_start:
{
lean_object* v___x_602_; 
v___x_602_ = lp_mathlib_Lean_throwErrorAt___at___00Mathlib_Tactic_simpIntroCore_spec__1___redArg(v_ref_593_, v_msg_594_, v___y_595_, v___y_596_, v___y_597_, v___y_598_, v___y_599_, v___y_600_);
return v___x_602_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Mathlib_Tactic_simpIntroCore_spec__1___boxed(lean_object* v_00_u03b1_603_, lean_object* v_ref_604_, lean_object* v_msg_605_, lean_object* v___y_606_, lean_object* v___y_607_, lean_object* v___y_608_, lean_object* v___y_609_, lean_object* v___y_610_, lean_object* v___y_611_, lean_object* v___y_612_){
_start:
{
lean_object* v_res_613_; 
v_res_613_ = lp_mathlib_Lean_throwErrorAt___at___00Mathlib_Tactic_simpIntroCore_spec__1(v_00_u03b1_603_, v_ref_604_, v_msg_605_, v___y_606_, v___y_607_, v___y_608_, v___y_609_, v___y_610_, v___y_611_);
lean_dec(v___y_611_);
lean_dec_ref(v___y_610_);
lean_dec(v___y_609_);
lean_dec_ref(v___y_608_);
lean_dec(v___y_607_);
lean_dec_ref(v___y_606_);
lean_dec(v_ref_604_);
return v_res_613_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_simpIntroCore_spec__1_spec__1(lean_object* v_00_u03b1_614_, lean_object* v_msg_615_, lean_object* v___y_616_, lean_object* v___y_617_, lean_object* v___y_618_, lean_object* v___y_619_, lean_object* v___y_620_, lean_object* v___y_621_){
_start:
{
lean_object* v___x_623_; 
v___x_623_ = lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_simpIntroCore_spec__1_spec__1___redArg(v_msg_615_, v___y_616_, v___y_617_, v___y_618_, v___y_619_, v___y_620_, v___y_621_);
return v___x_623_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_simpIntroCore_spec__1_spec__1___boxed(lean_object* v_00_u03b1_624_, lean_object* v_msg_625_, lean_object* v___y_626_, lean_object* v___y_627_, lean_object* v___y_628_, lean_object* v___y_629_, lean_object* v___y_630_, lean_object* v___y_631_, lean_object* v___y_632_){
_start:
{
lean_object* v_res_633_; 
v_res_633_ = lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_simpIntroCore_spec__1_spec__1(v_00_u03b1_624_, v_msg_625_, v___y_626_, v___y_627_, v___y_628_, v___y_629_, v___y_630_, v___y_631_);
lean_dec(v___y_631_);
lean_dec_ref(v___y_630_);
lean_dec(v___y_629_);
lean_dec_ref(v___y_628_);
lean_dec(v___y_627_);
lean_dec_ref(v___y_626_);
return v_res_633_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_simpIntroCore_spec__1_spec__1_spec__3(lean_object* v_msgData_634_, lean_object* v_macroStack_635_, lean_object* v___y_636_, lean_object* v___y_637_, lean_object* v___y_638_, lean_object* v___y_639_, lean_object* v___y_640_, lean_object* v___y_641_){
_start:
{
lean_object* v___x_643_; 
v___x_643_ = lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_simpIntroCore_spec__1_spec__1_spec__3___redArg(v_msgData_634_, v_macroStack_635_, v___y_640_);
return v___x_643_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_simpIntroCore_spec__1_spec__1_spec__3___boxed(lean_object* v_msgData_644_, lean_object* v_macroStack_645_, lean_object* v___y_646_, lean_object* v___y_647_, lean_object* v___y_648_, lean_object* v___y_649_, lean_object* v___y_650_, lean_object* v___y_651_, lean_object* v___y_652_){
_start:
{
lean_object* v_res_653_; 
v_res_653_ = lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Tactic_simpIntroCore_spec__1_spec__1_spec__3(v_msgData_644_, v_macroStack_645_, v___y_646_, v___y_647_, v___y_648_, v___y_649_, v___y_650_, v___y_651_);
lean_dec(v___y_651_);
lean_dec_ref(v___y_650_);
lean_dec(v___y_649_);
lean_dec_ref(v___y_648_);
lean_dec(v___y_647_);
lean_dec_ref(v___y_646_);
return v_res_653_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__8(void){
_start:
{
lean_object* v___x_668_; lean_object* v___x_669_; lean_object* v___x_670_; lean_object* v___x_671_; 
v___x_668_ = l_Lean_Parser_Tactic_optConfig;
v___x_669_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__7));
v___x_670_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__5));
v___x_671_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_671_, 0, v___x_670_);
lean_ctor_set(v___x_671_, 1, v___x_669_);
lean_ctor_set(v___x_671_, 2, v___x_668_);
return v___x_671_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__11(void){
_start:
{
lean_object* v___x_675_; lean_object* v___x_676_; lean_object* v___x_677_; 
v___x_675_ = l_Lean_Parser_Tactic_discharger;
v___x_676_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__10));
v___x_677_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_677_, 0, v___x_676_);
lean_ctor_set(v___x_677_, 1, v___x_675_);
return v___x_677_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__12(void){
_start:
{
lean_object* v___x_678_; lean_object* v___x_679_; lean_object* v___x_680_; lean_object* v___x_681_; 
v___x_678_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__11, &lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__11_once, _init_lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__11);
v___x_679_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__8, &lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__8_once, _init_lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__8);
v___x_680_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__5));
v___x_681_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_681_, 0, v___x_680_);
lean_ctor_set(v___x_681_, 1, v___x_679_);
lean_ctor_set(v___x_681_, 2, v___x_678_);
return v___x_681_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__22(void){
_start:
{
lean_object* v___x_699_; lean_object* v___x_700_; lean_object* v___x_701_; lean_object* v___x_702_; 
v___x_699_ = l_Lean_binderIdent;
v___x_700_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__21));
v___x_701_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__5));
v___x_702_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_702_, 0, v___x_701_);
lean_ctor_set(v___x_702_, 1, v___x_700_);
lean_ctor_set(v___x_702_, 2, v___x_699_);
return v___x_702_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__23(void){
_start:
{
lean_object* v___x_703_; lean_object* v___x_704_; lean_object* v___x_705_; 
v___x_703_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__22, &lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__22_once, _init_lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__22);
v___x_704_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__14));
v___x_705_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_705_, 0, v___x_704_);
lean_ctor_set(v___x_705_, 1, v___x_703_);
return v___x_705_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__24(void){
_start:
{
lean_object* v___x_706_; lean_object* v___x_707_; lean_object* v___x_708_; lean_object* v___x_709_; 
v___x_706_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__23, &lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__23_once, _init_lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__23);
v___x_707_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__12, &lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__12_once, _init_lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__12);
v___x_708_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__5));
v___x_709_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_709_, 0, v___x_708_);
lean_ctor_set(v___x_709_, 1, v___x_707_);
lean_ctor_set(v___x_709_, 2, v___x_706_);
return v___x_709_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__28(void){
_start:
{
lean_object* v___x_716_; lean_object* v___x_717_; lean_object* v___x_718_; lean_object* v___x_719_; 
v___x_716_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__27));
v___x_717_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__24, &lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__24_once, _init_lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__24);
v___x_718_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__5));
v___x_719_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_719_, 0, v___x_718_);
lean_ctor_set(v___x_719_, 1, v___x_717_);
lean_ctor_set(v___x_719_, 2, v___x_716_);
return v___x_719_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__32(void){
_start:
{
lean_object* v___x_727_; lean_object* v___x_728_; lean_object* v___x_729_; lean_object* v___x_730_; 
v___x_727_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__31));
v___x_728_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__28, &lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__28_once, _init_lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__28);
v___x_729_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__5));
v___x_730_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_730_, 0, v___x_729_);
lean_ctor_set(v___x_730_, 1, v___x_728_);
lean_ctor_set(v___x_730_, 2, v___x_727_);
return v___x_730_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__33(void){
_start:
{
lean_object* v___x_731_; lean_object* v___x_732_; lean_object* v___x_733_; 
v___x_731_ = l_Lean_Parser_Tactic_simpArgs;
v___x_732_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__10));
v___x_733_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_733_, 0, v___x_732_);
lean_ctor_set(v___x_733_, 1, v___x_731_);
return v___x_733_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__34(void){
_start:
{
lean_object* v___x_734_; lean_object* v___x_735_; lean_object* v___x_736_; lean_object* v___x_737_; 
v___x_734_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__33, &lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__33_once, _init_lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__33);
v___x_735_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__32, &lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__32_once, _init_lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__32);
v___x_736_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__5));
v___x_737_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_737_, 0, v___x_736_);
lean_ctor_set(v___x_737_, 1, v___x_735_);
lean_ctor_set(v___x_737_, 2, v___x_734_);
return v___x_737_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__35(void){
_start:
{
lean_object* v___x_738_; lean_object* v___x_739_; lean_object* v___x_740_; lean_object* v___x_741_; 
v___x_738_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__34, &lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__34_once, _init_lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__34);
v___x_739_ = lean_unsigned_to_nat(1022u);
v___x_740_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__3));
v___x_741_ = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(v___x_741_, 0, v___x_740_);
lean_ctor_set(v___x_741_, 1, v___x_739_);
lean_ctor_set(v___x_741_, 2, v___x_738_);
return v___x_741_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly__(void){
_start:
{
lean_object* v___x_742_; 
v___x_742_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__35, &lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__35_once, _init_lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__35);
return v___x_742_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SimpIntro______elabRules__Mathlib__Tactic__tacticSimp__intro___________x2e_x2eOnly____1_spec__0___redArg___closed__0(void){
_start:
{
lean_object* v___x_743_; lean_object* v___x_744_; lean_object* v___x_745_; 
v___x_743_ = lean_box(0);
v___x_744_ = l_Lean_Elab_unsupportedSyntaxExceptionId;
v___x_745_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_745_, 0, v___x_744_);
lean_ctor_set(v___x_745_, 1, v___x_743_);
return v___x_745_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SimpIntro______elabRules__Mathlib__Tactic__tacticSimp__intro___________x2e_x2eOnly____1_spec__0___redArg(){
_start:
{
lean_object* v___x_747_; lean_object* v___x_748_; 
v___x_747_ = lean_obj_once(&lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SimpIntro______elabRules__Mathlib__Tactic__tacticSimp__intro___________x2e_x2eOnly____1_spec__0___redArg___closed__0, &lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SimpIntro______elabRules__Mathlib__Tactic__tacticSimp__intro___________x2e_x2eOnly____1_spec__0___redArg___closed__0_once, _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SimpIntro______elabRules__Mathlib__Tactic__tacticSimp__intro___________x2e_x2eOnly____1_spec__0___redArg___closed__0);
v___x_748_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_748_, 0, v___x_747_);
return v___x_748_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SimpIntro______elabRules__Mathlib__Tactic__tacticSimp__intro___________x2e_x2eOnly____1_spec__0___redArg___boxed(lean_object* v___y_749_){
_start:
{
lean_object* v_res_750_; 
v_res_750_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SimpIntro______elabRules__Mathlib__Tactic__tacticSimp__intro___________x2e_x2eOnly____1_spec__0___redArg();
return v_res_750_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SimpIntro______elabRules__Mathlib__Tactic__tacticSimp__intro___________x2e_x2eOnly____1_spec__0(lean_object* v_00_u03b1_751_, lean_object* v___y_752_, lean_object* v___y_753_, lean_object* v___y_754_, lean_object* v___y_755_, lean_object* v___y_756_, lean_object* v___y_757_, lean_object* v___y_758_, lean_object* v___y_759_){
_start:
{
lean_object* v___x_761_; 
v___x_761_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SimpIntro______elabRules__Mathlib__Tactic__tacticSimp__intro___________x2e_x2eOnly____1_spec__0___redArg();
return v___x_761_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SimpIntro______elabRules__Mathlib__Tactic__tacticSimp__intro___________x2e_x2eOnly____1_spec__0___boxed(lean_object* v_00_u03b1_762_, lean_object* v___y_763_, lean_object* v___y_764_, lean_object* v___y_765_, lean_object* v___y_766_, lean_object* v___y_767_, lean_object* v___y_768_, lean_object* v___y_769_, lean_object* v___y_770_, lean_object* v___y_771_){
_start:
{
lean_object* v_res_772_; 
v_res_772_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SimpIntro______elabRules__Mathlib__Tactic__tacticSimp__intro___________x2e_x2eOnly____1_spec__0(v_00_u03b1_762_, v___y_763_, v___y_764_, v___y_765_, v___y_766_, v___y_767_, v___y_768_, v___y_769_, v___y_770_);
lean_dec(v___y_770_);
lean_dec_ref(v___y_769_);
lean_dec(v___y_768_);
lean_dec_ref(v___y_767_);
lean_dec(v___y_766_);
lean_dec_ref(v___y_765_);
lean_dec(v___y_764_);
lean_dec_ref(v___y_763_);
return v_res_772_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SimpIntro______elabRules__Mathlib__Tactic__tacticSimp__intro___________x2e_x2eOnly____1_spec__1___redArg___lam__0(lean_object* v_x_773_, lean_object* v___y_774_, lean_object* v___y_775_, lean_object* v___y_776_, lean_object* v___y_777_, lean_object* v___y_778_, lean_object* v___y_779_, lean_object* v___y_780_, lean_object* v___y_781_){
_start:
{
lean_object* v___x_783_; 
lean_inc(v___y_777_);
lean_inc_ref(v___y_776_);
lean_inc(v___y_775_);
lean_inc_ref(v___y_774_);
v___x_783_ = lean_apply_9(v_x_773_, v___y_774_, v___y_775_, v___y_776_, v___y_777_, v___y_778_, v___y_779_, v___y_780_, v___y_781_, lean_box(0));
return v___x_783_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SimpIntro______elabRules__Mathlib__Tactic__tacticSimp__intro___________x2e_x2eOnly____1_spec__1___redArg___lam__0___boxed(lean_object* v_x_784_, lean_object* v___y_785_, lean_object* v___y_786_, lean_object* v___y_787_, lean_object* v___y_788_, lean_object* v___y_789_, lean_object* v___y_790_, lean_object* v___y_791_, lean_object* v___y_792_, lean_object* v___y_793_){
_start:
{
lean_object* v_res_794_; 
v_res_794_ = lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SimpIntro______elabRules__Mathlib__Tactic__tacticSimp__intro___________x2e_x2eOnly____1_spec__1___redArg___lam__0(v_x_784_, v___y_785_, v___y_786_, v___y_787_, v___y_788_, v___y_789_, v___y_790_, v___y_791_, v___y_792_);
lean_dec(v___y_788_);
lean_dec_ref(v___y_787_);
lean_dec(v___y_786_);
lean_dec_ref(v___y_785_);
return v_res_794_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SimpIntro______elabRules__Mathlib__Tactic__tacticSimp__intro___________x2e_x2eOnly____1_spec__1___redArg(lean_object* v_mvarId_795_, lean_object* v_x_796_, lean_object* v___y_797_, lean_object* v___y_798_, lean_object* v___y_799_, lean_object* v___y_800_, lean_object* v___y_801_, lean_object* v___y_802_, lean_object* v___y_803_, lean_object* v___y_804_){
_start:
{
lean_object* v___f_806_; lean_object* v___x_807_; 
lean_inc(v___y_800_);
lean_inc_ref(v___y_799_);
lean_inc(v___y_798_);
lean_inc_ref(v___y_797_);
v___f_806_ = lean_alloc_closure((void*)(lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SimpIntro______elabRules__Mathlib__Tactic__tacticSimp__intro___________x2e_x2eOnly____1_spec__1___redArg___lam__0___boxed), 10, 5);
lean_closure_set(v___f_806_, 0, v_x_796_);
lean_closure_set(v___f_806_, 1, v___y_797_);
lean_closure_set(v___f_806_, 2, v___y_798_);
lean_closure_set(v___f_806_, 3, v___y_799_);
lean_closure_set(v___f_806_, 4, v___y_800_);
v___x_807_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withMVarContextImp(lean_box(0), v_mvarId_795_, v___f_806_, v___y_801_, v___y_802_, v___y_803_, v___y_804_);
if (lean_obj_tag(v___x_807_) == 0)
{
return v___x_807_;
}
else
{
lean_object* v_a_808_; lean_object* v___x_810_; uint8_t v_isShared_811_; uint8_t v_isSharedCheck_815_; 
v_a_808_ = lean_ctor_get(v___x_807_, 0);
v_isSharedCheck_815_ = !lean_is_exclusive(v___x_807_);
if (v_isSharedCheck_815_ == 0)
{
v___x_810_ = v___x_807_;
v_isShared_811_ = v_isSharedCheck_815_;
goto v_resetjp_809_;
}
else
{
lean_inc(v_a_808_);
lean_dec(v___x_807_);
v___x_810_ = lean_box(0);
v_isShared_811_ = v_isSharedCheck_815_;
goto v_resetjp_809_;
}
v_resetjp_809_:
{
lean_object* v___x_813_; 
if (v_isShared_811_ == 0)
{
v___x_813_ = v___x_810_;
goto v_reusejp_812_;
}
else
{
lean_object* v_reuseFailAlloc_814_; 
v_reuseFailAlloc_814_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_814_, 0, v_a_808_);
v___x_813_ = v_reuseFailAlloc_814_;
goto v_reusejp_812_;
}
v_reusejp_812_:
{
return v___x_813_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SimpIntro______elabRules__Mathlib__Tactic__tacticSimp__intro___________x2e_x2eOnly____1_spec__1___redArg___boxed(lean_object* v_mvarId_816_, lean_object* v_x_817_, lean_object* v___y_818_, lean_object* v___y_819_, lean_object* v___y_820_, lean_object* v___y_821_, lean_object* v___y_822_, lean_object* v___y_823_, lean_object* v___y_824_, lean_object* v___y_825_, lean_object* v___y_826_){
_start:
{
lean_object* v_res_827_; 
v_res_827_ = lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SimpIntro______elabRules__Mathlib__Tactic__tacticSimp__intro___________x2e_x2eOnly____1_spec__1___redArg(v_mvarId_816_, v_x_817_, v___y_818_, v___y_819_, v___y_820_, v___y_821_, v___y_822_, v___y_823_, v___y_824_, v___y_825_);
lean_dec(v___y_825_);
lean_dec_ref(v___y_824_);
lean_dec(v___y_823_);
lean_dec_ref(v___y_822_);
lean_dec(v___y_821_);
lean_dec_ref(v___y_820_);
lean_dec(v___y_819_);
lean_dec_ref(v___y_818_);
return v_res_827_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SimpIntro______elabRules__Mathlib__Tactic__tacticSimp__intro___________x2e_x2eOnly____1_spec__1(lean_object* v_00_u03b1_828_, lean_object* v_mvarId_829_, lean_object* v_x_830_, lean_object* v___y_831_, lean_object* v___y_832_, lean_object* v___y_833_, lean_object* v___y_834_, lean_object* v___y_835_, lean_object* v___y_836_, lean_object* v___y_837_, lean_object* v___y_838_){
_start:
{
lean_object* v___x_840_; 
v___x_840_ = lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SimpIntro______elabRules__Mathlib__Tactic__tacticSimp__intro___________x2e_x2eOnly____1_spec__1___redArg(v_mvarId_829_, v_x_830_, v___y_831_, v___y_832_, v___y_833_, v___y_834_, v___y_835_, v___y_836_, v___y_837_, v___y_838_);
return v___x_840_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SimpIntro______elabRules__Mathlib__Tactic__tacticSimp__intro___________x2e_x2eOnly____1_spec__1___boxed(lean_object* v_00_u03b1_841_, lean_object* v_mvarId_842_, lean_object* v_x_843_, lean_object* v___y_844_, lean_object* v___y_845_, lean_object* v___y_846_, lean_object* v___y_847_, lean_object* v___y_848_, lean_object* v___y_849_, lean_object* v___y_850_, lean_object* v___y_851_, lean_object* v___y_852_){
_start:
{
lean_object* v_res_853_; 
v_res_853_ = lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SimpIntro______elabRules__Mathlib__Tactic__tacticSimp__intro___________x2e_x2eOnly____1_spec__1(v_00_u03b1_841_, v_mvarId_842_, v_x_843_, v___y_844_, v___y_845_, v___y_846_, v___y_847_, v___y_848_, v___y_849_, v___y_850_, v___y_851_);
lean_dec(v___y_851_);
lean_dec_ref(v___y_850_);
lean_dec(v___y_849_);
lean_dec_ref(v___y_848_);
lean_dec(v___y_847_);
lean_dec_ref(v___y_846_);
lean_dec(v___y_845_);
lean_dec_ref(v___y_844_);
return v_res_853_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpIntro______elabRules__Mathlib__Tactic__tacticSimp__intro___________x2e_x2eOnly____1___lam__0(lean_object* v_a_854_, lean_object* v_ctx_855_, lean_object* v_simprocs_856_, lean_object* v_discharge_x3f_857_, uint8_t v___y_858_, lean_object* v___x_859_, lean_object* v___y_860_, lean_object* v___y_861_, lean_object* v___y_862_, lean_object* v___y_863_, lean_object* v___y_864_, lean_object* v___y_865_, lean_object* v___y_866_, lean_object* v___y_867_){
_start:
{
lean_object* v___x_869_; 
v___x_869_ = lp_mathlib_Mathlib_Tactic_simpIntroCore(v_a_854_, v_ctx_855_, v_simprocs_856_, v_discharge_x3f_857_, v___y_858_, v___x_859_, v___y_862_, v___y_863_, v___y_864_, v___y_865_, v___y_866_, v___y_867_);
if (lean_obj_tag(v___x_869_) == 0)
{
lean_object* v_a_870_; 
v_a_870_ = lean_ctor_get(v___x_869_, 0);
lean_inc(v_a_870_);
lean_dec_ref_known(v___x_869_, 1);
if (lean_obj_tag(v_a_870_) == 1)
{
lean_object* v_val_871_; lean_object* v___x_872_; lean_object* v___x_873_; lean_object* v___x_874_; 
v_val_871_ = lean_ctor_get(v_a_870_, 0);
lean_inc(v_val_871_);
lean_dec_ref_known(v_a_870_, 1);
v___x_872_ = lean_box(0);
v___x_873_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_873_, 0, v_val_871_);
lean_ctor_set(v___x_873_, 1, v___x_872_);
v___x_874_ = l_Lean_Elab_Tactic_replaceMainGoal___redArg(v___x_873_, v___y_861_, v___y_864_, v___y_865_, v___y_866_, v___y_867_);
return v___x_874_;
}
else
{
lean_object* v___x_875_; lean_object* v___x_876_; 
lean_dec(v_a_870_);
v___x_875_ = lean_box(0);
v___x_876_ = l_Lean_Elab_Tactic_replaceMainGoal___redArg(v___x_875_, v___y_861_, v___y_864_, v___y_865_, v___y_866_, v___y_867_);
return v___x_876_;
}
}
else
{
lean_object* v_a_877_; lean_object* v___x_879_; uint8_t v_isShared_880_; uint8_t v_isSharedCheck_884_; 
v_a_877_ = lean_ctor_get(v___x_869_, 0);
v_isSharedCheck_884_ = !lean_is_exclusive(v___x_869_);
if (v_isSharedCheck_884_ == 0)
{
v___x_879_ = v___x_869_;
v_isShared_880_ = v_isSharedCheck_884_;
goto v_resetjp_878_;
}
else
{
lean_inc(v_a_877_);
lean_dec(v___x_869_);
v___x_879_ = lean_box(0);
v_isShared_880_ = v_isSharedCheck_884_;
goto v_resetjp_878_;
}
v_resetjp_878_:
{
lean_object* v___x_882_; 
if (v_isShared_880_ == 0)
{
v___x_882_ = v___x_879_;
goto v_reusejp_881_;
}
else
{
lean_object* v_reuseFailAlloc_883_; 
v_reuseFailAlloc_883_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_883_, 0, v_a_877_);
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
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpIntro______elabRules__Mathlib__Tactic__tacticSimp__intro___________x2e_x2eOnly____1___lam__0___boxed(lean_object* v_a_885_, lean_object* v_ctx_886_, lean_object* v_simprocs_887_, lean_object* v_discharge_x3f_888_, lean_object* v___y_889_, lean_object* v___x_890_, lean_object* v___y_891_, lean_object* v___y_892_, lean_object* v___y_893_, lean_object* v___y_894_, lean_object* v___y_895_, lean_object* v___y_896_, lean_object* v___y_897_, lean_object* v___y_898_, lean_object* v___y_899_){
_start:
{
uint8_t v___y_6325__boxed_900_; lean_object* v_res_901_; 
v___y_6325__boxed_900_ = lean_unbox(v___y_889_);
v_res_901_ = lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpIntro______elabRules__Mathlib__Tactic__tacticSimp__intro___________x2e_x2eOnly____1___lam__0(v_a_885_, v_ctx_886_, v_simprocs_887_, v_discharge_x3f_888_, v___y_6325__boxed_900_, v___x_890_, v___y_891_, v___y_892_, v___y_893_, v___y_894_, v___y_895_, v___y_896_, v___y_897_, v___y_898_);
lean_dec(v___y_898_);
lean_dec_ref(v___y_897_);
lean_dec(v___y_896_);
lean_dec_ref(v___y_895_);
lean_dec(v___y_894_);
lean_dec_ref(v___y_893_);
lean_dec(v___y_892_);
lean_dec_ref(v___y_891_);
return v_res_901_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpIntro______elabRules__Mathlib__Tactic__tacticSimp__intro___________x2e_x2eOnly____1___lam__1(lean_object* v_ids_904_, lean_object* v_ctx_905_, lean_object* v_simprocs_906_, lean_object* v___y_907_, uint8_t v___x_908_, uint8_t v___x_909_, lean_object* v_discharge_x3f_910_, lean_object* v___y_911_, lean_object* v___y_912_, lean_object* v___y_913_, lean_object* v___y_914_, lean_object* v___y_915_, lean_object* v___y_916_, lean_object* v___y_917_, lean_object* v___y_918_){
_start:
{
lean_object* v___x_920_; 
v___x_920_ = l_Lean_Elab_Tactic_getMainGoal___redArg(v___y_912_, v___y_915_, v___y_916_, v___y_917_, v___y_918_);
if (lean_obj_tag(v___x_920_) == 0)
{
lean_object* v_a_921_; lean_object* v___x_922_; lean_object* v___x_923_; 
v_a_921_ = lean_ctor_get(v___x_920_, 0);
lean_inc_n(v_a_921_, 2);
lean_dec_ref_known(v___x_920_, 1);
v___x_922_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpIntro______elabRules__Mathlib__Tactic__tacticSimp__intro___________x2e_x2eOnly____1___lam__1___closed__0));
v___x_923_ = l_Lean_MVarId_checkNotAssigned(v_a_921_, v___x_922_, v___y_915_, v___y_916_, v___y_917_, v___y_918_);
if (lean_obj_tag(v___x_923_) == 0)
{
uint8_t v___y_925_; 
lean_dec_ref_known(v___x_923_, 1);
if (lean_obj_tag(v___y_907_) == 0)
{
v___y_925_ = v___x_908_;
goto v___jp_924_;
}
else
{
v___y_925_ = v___x_909_;
goto v___jp_924_;
}
v___jp_924_:
{
lean_object* v___x_926_; lean_object* v___x_927_; lean_object* v___f_928_; lean_object* v___x_929_; 
v___x_926_ = lean_array_to_list(v_ids_904_);
v___x_927_ = lean_box(v___y_925_);
lean_inc(v_a_921_);
v___f_928_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpIntro______elabRules__Mathlib__Tactic__tacticSimp__intro___________x2e_x2eOnly____1___lam__0___boxed), 15, 6);
lean_closure_set(v___f_928_, 0, v_a_921_);
lean_closure_set(v___f_928_, 1, v_ctx_905_);
lean_closure_set(v___f_928_, 2, v_simprocs_906_);
lean_closure_set(v___f_928_, 3, v_discharge_x3f_910_);
lean_closure_set(v___f_928_, 4, v___x_927_);
lean_closure_set(v___f_928_, 5, v___x_926_);
v___x_929_ = lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SimpIntro______elabRules__Mathlib__Tactic__tacticSimp__intro___________x2e_x2eOnly____1_spec__1___redArg(v_a_921_, v___f_928_, v___y_911_, v___y_912_, v___y_913_, v___y_914_, v___y_915_, v___y_916_, v___y_917_, v___y_918_);
return v___x_929_;
}
}
else
{
lean_dec(v_a_921_);
lean_dec(v_discharge_x3f_910_);
lean_dec_ref(v_simprocs_906_);
lean_dec_ref(v_ctx_905_);
lean_dec_ref(v_ids_904_);
return v___x_923_;
}
}
else
{
lean_object* v_a_930_; lean_object* v___x_932_; uint8_t v_isShared_933_; uint8_t v_isSharedCheck_937_; 
lean_dec(v_discharge_x3f_910_);
lean_dec_ref(v_simprocs_906_);
lean_dec_ref(v_ctx_905_);
lean_dec_ref(v_ids_904_);
v_a_930_ = lean_ctor_get(v___x_920_, 0);
v_isSharedCheck_937_ = !lean_is_exclusive(v___x_920_);
if (v_isSharedCheck_937_ == 0)
{
v___x_932_ = v___x_920_;
v_isShared_933_ = v_isSharedCheck_937_;
goto v_resetjp_931_;
}
else
{
lean_inc(v_a_930_);
lean_dec(v___x_920_);
v___x_932_ = lean_box(0);
v_isShared_933_ = v_isSharedCheck_937_;
goto v_resetjp_931_;
}
v_resetjp_931_:
{
lean_object* v___x_935_; 
if (v_isShared_933_ == 0)
{
v___x_935_ = v___x_932_;
goto v_reusejp_934_;
}
else
{
lean_object* v_reuseFailAlloc_936_; 
v_reuseFailAlloc_936_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_936_, 0, v_a_930_);
v___x_935_ = v_reuseFailAlloc_936_;
goto v_reusejp_934_;
}
v_reusejp_934_:
{
return v___x_935_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpIntro______elabRules__Mathlib__Tactic__tacticSimp__intro___________x2e_x2eOnly____1___lam__1___boxed(lean_object* v_ids_938_, lean_object* v_ctx_939_, lean_object* v_simprocs_940_, lean_object* v___y_941_, lean_object* v___x_942_, lean_object* v___x_943_, lean_object* v_discharge_x3f_944_, lean_object* v___y_945_, lean_object* v___y_946_, lean_object* v___y_947_, lean_object* v___y_948_, lean_object* v___y_949_, lean_object* v___y_950_, lean_object* v___y_951_, lean_object* v___y_952_, lean_object* v___y_953_){
_start:
{
uint8_t v___x_6404__boxed_954_; uint8_t v___x_6405__boxed_955_; lean_object* v_res_956_; 
v___x_6404__boxed_954_ = lean_unbox(v___x_942_);
v___x_6405__boxed_955_ = lean_unbox(v___x_943_);
v_res_956_ = lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpIntro______elabRules__Mathlib__Tactic__tacticSimp__intro___________x2e_x2eOnly____1___lam__1(v_ids_938_, v_ctx_939_, v_simprocs_940_, v___y_941_, v___x_6404__boxed_954_, v___x_6405__boxed_955_, v_discharge_x3f_944_, v___y_945_, v___y_946_, v___y_947_, v___y_948_, v___y_949_, v___y_950_, v___y_951_, v___y_952_);
lean_dec(v___y_952_);
lean_dec_ref(v___y_951_);
lean_dec(v___y_950_);
lean_dec_ref(v___y_949_);
lean_dec(v___y_948_);
lean_dec_ref(v___y_947_);
lean_dec(v___y_946_);
lean_dec_ref(v___y_945_);
lean_dec(v___y_941_);
return v_res_956_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpIntro______elabRules__Mathlib__Tactic__tacticSimp__intro___________x2e_x2eOnly____1___closed__8(void){
_start:
{
lean_object* v___x_967_; 
v___x_967_ = l_Array_mkArray0(lean_box(0));
return v___x_967_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpIntro______elabRules__Mathlib__Tactic__tacticSimp__intro___________x2e_x2eOnly____1(lean_object* v_x_970_, lean_object* v_a_971_, lean_object* v_a_972_, lean_object* v_a_973_, lean_object* v_a_974_, lean_object* v_a_975_, lean_object* v_a_976_, lean_object* v_a_977_, lean_object* v_a_978_){
_start:
{
lean_object* v___x_980_; lean_object* v___x_981_; uint8_t v___x_982_; 
v___x_980_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__1));
v___x_981_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly___00__closed__3));
lean_inc(v_x_970_);
v___x_982_ = l_Lean_Syntax_isOfKind(v_x_970_, v___x_981_);
if (v___x_982_ == 0)
{
lean_object* v___x_983_; 
lean_dec(v_x_970_);
v___x_983_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__SimpIntro______elabRules__Mathlib__Tactic__tacticSimp__intro___________x2e_x2eOnly____1_spec__0___redArg();
return v___x_983_;
}
else
{
lean_object* v___x_984_; lean_object* v___x_985_; lean_object* v___y_987_; uint8_t v___y_988_; lean_object* v___y_989_; lean_object* v___y_990_; uint8_t v___y_991_; lean_object* v___y_992_; lean_object* v___y_993_; lean_object* v___y_994_; lean_object* v___y_995_; lean_object* v___y_996_; lean_object* v___y_997_; lean_object* v___y_998_; lean_object* v___y_1027_; uint8_t v___y_1028_; lean_object* v___y_1029_; lean_object* v___y_1030_; uint8_t v___y_1031_; lean_object* v___y_1032_; lean_object* v___y_1033_; lean_object* v___y_1034_; lean_object* v___y_1035_; lean_object* v___y_1036_; lean_object* v___y_1037_; lean_object* v___y_1038_; lean_object* v___y_1051_; uint8_t v___y_1052_; lean_object* v___y_1053_; lean_object* v___y_1054_; lean_object* v___y_1055_; lean_object* v___y_1056_; lean_object* v___y_1057_; lean_object* v___y_1058_; lean_object* v___y_1059_; lean_object* v___y_1060_; lean_object* v___y_1061_; lean_object* v___y_1071_; lean_object* v___y_1072_; lean_object* v___y_1073_; lean_object* v___y_1074_; lean_object* v___y_1075_; lean_object* v___y_1076_; lean_object* v___y_1077_; lean_object* v___y_1091_; lean_object* v___y_1092_; lean_object* v___y_1093_; lean_object* v___y_1094_; lean_object* v___y_1095_; lean_object* v___y_1096_; lean_object* v___x_1109_; lean_object* v___x_1110_; lean_object* v___x_1111_; lean_object* v___x_1112_; lean_object* v___y_1114_; lean_object* v___y_1115_; lean_object* v___y_1116_; lean_object* v___x_1129_; lean_object* v___x_1130_; lean_object* v___y_1132_; lean_object* v___y_1133_; lean_object* v___x_1144_; lean_object* v___x_1145_; lean_object* v___y_1147_; lean_object* v___x_1158_; lean_object* v___x_1159_; lean_object* v___x_1160_; 
v___x_984_ = lean_unsigned_to_nat(1u);
v___x_985_ = l_Lean_Syntax_getArg(v_x_970_, v___x_984_);
v___x_1109_ = lean_unsigned_to_nat(2u);
v___x_1110_ = l_Lean_Syntax_getArg(v_x_970_, v___x_1109_);
v___x_1111_ = lean_unsigned_to_nat(3u);
v___x_1112_ = l_Lean_Syntax_getArg(v_x_970_, v___x_1111_);
v___x_1129_ = lean_unsigned_to_nat(4u);
v___x_1130_ = l_Lean_Syntax_getArg(v_x_970_, v___x_1129_);
v___x_1144_ = lean_unsigned_to_nat(5u);
v___x_1145_ = l_Lean_Syntax_getArg(v_x_970_, v___x_1144_);
v___x_1158_ = lean_unsigned_to_nat(6u);
v___x_1159_ = l_Lean_Syntax_getArg(v_x_970_, v___x_1158_);
lean_dec(v_x_970_);
v___x_1160_ = l_Lean_Syntax_getOptional_x3f(v___x_1159_);
lean_dec(v___x_1159_);
if (lean_obj_tag(v___x_1160_) == 0)
{
lean_object* v___x_1161_; 
v___x_1161_ = lean_box(0);
v___y_1147_ = v___x_1161_;
goto v___jp_1146_;
}
else
{
lean_object* v_val_1162_; lean_object* v___x_1164_; uint8_t v_isShared_1165_; uint8_t v_isSharedCheck_1169_; 
v_val_1162_ = lean_ctor_get(v___x_1160_, 0);
v_isSharedCheck_1169_ = !lean_is_exclusive(v___x_1160_);
if (v_isSharedCheck_1169_ == 0)
{
v___x_1164_ = v___x_1160_;
v_isShared_1165_ = v_isSharedCheck_1169_;
goto v_resetjp_1163_;
}
else
{
lean_inc(v_val_1162_);
lean_dec(v___x_1160_);
v___x_1164_ = lean_box(0);
v_isShared_1165_ = v_isSharedCheck_1169_;
goto v_resetjp_1163_;
}
v_resetjp_1163_:
{
lean_object* v___x_1167_; 
if (v_isShared_1165_ == 0)
{
v___x_1167_ = v___x_1164_;
goto v_reusejp_1166_;
}
else
{
lean_object* v_reuseFailAlloc_1168_; 
v_reuseFailAlloc_1168_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1168_, 0, v_val_1162_);
v___x_1167_ = v_reuseFailAlloc_1168_;
goto v_reusejp_1166_;
}
v_reusejp_1166_:
{
v___y_1147_ = v___x_1167_;
goto v___jp_1146_;
}
}
}
v___jp_986_:
{
lean_object* v___x_999_; lean_object* v___x_1000_; lean_object* v___x_1001_; lean_object* v___x_1002_; uint8_t v___x_1003_; lean_object* v___x_1004_; lean_object* v___x_1005_; lean_object* v___x_1006_; lean_object* v___x_1007_; lean_object* v___x_1008_; lean_object* v___x_1009_; 
lean_inc_ref_n(v___y_996_, 2);
v___x_999_ = l_Array_append___redArg(v___y_996_, v___y_998_);
lean_dec_ref(v___y_998_);
lean_inc_n(v___y_990_, 2);
lean_inc_n(v___y_995_, 2);
v___x_1000_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_1000_, 0, v___y_995_);
lean_ctor_set(v___x_1000_, 1, v___y_990_);
lean_ctor_set(v___x_1000_, 2, v___x_999_);
v___x_1001_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_1001_, 0, v___y_995_);
lean_ctor_set(v___x_1001_, 1, v___y_990_);
lean_ctor_set(v___x_1001_, 2, v___y_996_);
v___x_1002_ = l_Lean_Syntax_node6(v___y_995_, v___y_992_, v___y_993_, v___x_985_, v___y_994_, v___y_997_, v___x_1000_, v___x_1001_);
v___x_1003_ = 0;
v___x_1004_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpIntro______elabRules__Mathlib__Tactic__tacticSimp__intro___________x2e_x2eOnly____1___closed__0));
v___x_1005_ = lean_box(v___y_991_);
v___x_1006_ = lean_box(v___x_1003_);
v___x_1007_ = lean_box(v___y_991_);
v___x_1008_ = lean_alloc_closure((void*)(l_Lean_Elab_Tactic_mkSimpContext___boxed), 14, 5);
lean_closure_set(v___x_1008_, 0, v___x_1002_);
lean_closure_set(v___x_1008_, 1, v___x_1005_);
lean_closure_set(v___x_1008_, 2, v___x_1006_);
lean_closure_set(v___x_1008_, 3, v___x_1007_);
lean_closure_set(v___x_1008_, 4, v___x_1004_);
v___x_1009_ = l_Lean_Elab_Tactic_withMainContext___redArg(v___x_1008_, v_a_971_, v_a_972_, v_a_973_, v_a_974_, v_a_975_, v_a_976_, v_a_977_, v_a_978_);
if (lean_obj_tag(v___x_1009_) == 0)
{
lean_object* v_a_1010_; lean_object* v_ctx_1011_; lean_object* v_simprocs_1012_; lean_object* v_dischargeWrapper_1013_; lean_object* v___x_1014_; lean_object* v___x_1015_; lean_object* v___f_1016_; lean_object* v___x_1017_; 
v_a_1010_ = lean_ctor_get(v___x_1009_, 0);
lean_inc(v_a_1010_);
lean_dec_ref_known(v___x_1009_, 1);
v_ctx_1011_ = lean_ctor_get(v_a_1010_, 0);
lean_inc_ref(v_ctx_1011_);
v_simprocs_1012_ = lean_ctor_get(v_a_1010_, 1);
lean_inc_ref(v_simprocs_1012_);
v_dischargeWrapper_1013_ = lean_ctor_get(v_a_1010_, 2);
lean_inc(v_dischargeWrapper_1013_);
lean_dec(v_a_1010_);
v___x_1014_ = lean_box(v___y_988_);
v___x_1015_ = lean_box(v___x_982_);
v___f_1016_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpIntro______elabRules__Mathlib__Tactic__tacticSimp__intro___________x2e_x2eOnly____1___lam__1___boxed), 16, 6);
lean_closure_set(v___f_1016_, 0, v___y_989_);
lean_closure_set(v___f_1016_, 1, v_ctx_1011_);
lean_closure_set(v___f_1016_, 2, v_simprocs_1012_);
lean_closure_set(v___f_1016_, 3, v___y_987_);
lean_closure_set(v___f_1016_, 4, v___x_1014_);
lean_closure_set(v___f_1016_, 5, v___x_1015_);
v___x_1017_ = l_Lean_Elab_Tactic_Simp_DischargeWrapper_with___redArg(v_dischargeWrapper_1013_, v___f_1016_, v_a_971_, v_a_972_, v_a_973_, v_a_974_, v_a_975_, v_a_976_, v_a_977_, v_a_978_);
lean_dec(v_dischargeWrapper_1013_);
return v___x_1017_;
}
else
{
lean_object* v_a_1018_; lean_object* v___x_1020_; uint8_t v_isShared_1021_; uint8_t v_isSharedCheck_1025_; 
lean_dec_ref(v___y_989_);
lean_dec(v___y_987_);
v_a_1018_ = lean_ctor_get(v___x_1009_, 0);
v_isSharedCheck_1025_ = !lean_is_exclusive(v___x_1009_);
if (v_isSharedCheck_1025_ == 0)
{
v___x_1020_ = v___x_1009_;
v_isShared_1021_ = v_isSharedCheck_1025_;
goto v_resetjp_1019_;
}
else
{
lean_inc(v_a_1018_);
lean_dec(v___x_1009_);
v___x_1020_ = lean_box(0);
v_isShared_1021_ = v_isSharedCheck_1025_;
goto v_resetjp_1019_;
}
v_resetjp_1019_:
{
lean_object* v___x_1023_; 
if (v_isShared_1021_ == 0)
{
v___x_1023_ = v___x_1020_;
goto v_reusejp_1022_;
}
else
{
lean_object* v_reuseFailAlloc_1024_; 
v_reuseFailAlloc_1024_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1024_, 0, v_a_1018_);
v___x_1023_ = v_reuseFailAlloc_1024_;
goto v_reusejp_1022_;
}
v_reusejp_1022_:
{
return v___x_1023_;
}
}
}
}
v___jp_1026_:
{
lean_object* v___x_1039_; lean_object* v___x_1040_; 
lean_inc_ref(v___y_1036_);
v___x_1039_ = l_Array_append___redArg(v___y_1036_, v___y_1038_);
lean_dec_ref(v___y_1038_);
lean_inc(v___y_1030_);
lean_inc(v___y_1035_);
v___x_1040_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_1040_, 0, v___y_1035_);
lean_ctor_set(v___x_1040_, 1, v___y_1030_);
lean_ctor_set(v___x_1040_, 2, v___x_1039_);
if (lean_obj_tag(v___y_1037_) == 1)
{
lean_object* v_val_1041_; lean_object* v___x_1042_; lean_object* v___x_1043_; lean_object* v___x_1044_; lean_object* v___x_1045_; lean_object* v___x_1046_; lean_object* v___x_1047_; lean_object* v___x_1048_; 
v_val_1041_ = lean_ctor_get(v___y_1037_, 0);
lean_inc(v_val_1041_);
lean_dec_ref_known(v___y_1037_, 1);
v___x_1042_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpIntro______elabRules__Mathlib__Tactic__tacticSimp__intro___________x2e_x2eOnly____1___closed__1));
lean_inc_n(v___y_1035_, 3);
v___x_1043_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1043_, 0, v___y_1035_);
lean_ctor_set(v___x_1043_, 1, v___x_1042_);
lean_inc_ref(v___y_1036_);
v___x_1044_ = l_Array_append___redArg(v___y_1036_, v_val_1041_);
lean_dec(v_val_1041_);
lean_inc(v___y_1030_);
v___x_1045_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_1045_, 0, v___y_1035_);
lean_ctor_set(v___x_1045_, 1, v___y_1030_);
lean_ctor_set(v___x_1045_, 2, v___x_1044_);
v___x_1046_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpIntro______elabRules__Mathlib__Tactic__tacticSimp__intro___________x2e_x2eOnly____1___closed__2));
v___x_1047_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1047_, 0, v___y_1035_);
lean_ctor_set(v___x_1047_, 1, v___x_1046_);
v___x_1048_ = l_Array_mkArray3___redArg(v___x_1043_, v___x_1045_, v___x_1047_);
v___y_987_ = v___y_1027_;
v___y_988_ = v___y_1028_;
v___y_989_ = v___y_1029_;
v___y_990_ = v___y_1030_;
v___y_991_ = v___y_1031_;
v___y_992_ = v___y_1032_;
v___y_993_ = v___y_1033_;
v___y_994_ = v___y_1034_;
v___y_995_ = v___y_1035_;
v___y_996_ = v___y_1036_;
v___y_997_ = v___x_1040_;
v___y_998_ = v___x_1048_;
goto v___jp_986_;
}
else
{
lean_object* v___x_1049_; 
lean_dec(v___y_1037_);
v___x_1049_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpIntro______elabRules__Mathlib__Tactic__tacticSimp__intro___________x2e_x2eOnly____1___closed__3));
v___y_987_ = v___y_1027_;
v___y_988_ = v___y_1028_;
v___y_989_ = v___y_1029_;
v___y_990_ = v___y_1030_;
v___y_991_ = v___y_1031_;
v___y_992_ = v___y_1032_;
v___y_993_ = v___y_1033_;
v___y_994_ = v___y_1034_;
v___y_995_ = v___y_1035_;
v___y_996_ = v___y_1036_;
v___y_997_ = v___x_1040_;
v___y_998_ = v___x_1049_;
goto v___jp_986_;
}
}
v___jp_1050_:
{
lean_object* v___x_1062_; lean_object* v___x_1063_; 
lean_inc_ref(v___y_1058_);
v___x_1062_ = l_Array_append___redArg(v___y_1058_, v___y_1061_);
lean_dec_ref(v___y_1061_);
lean_inc(v___y_1054_);
lean_inc(v___y_1057_);
v___x_1063_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_1063_, 0, v___y_1057_);
lean_ctor_set(v___x_1063_, 1, v___y_1054_);
lean_ctor_set(v___x_1063_, 2, v___x_1062_);
if (lean_obj_tag(v___y_1059_) == 1)
{
lean_object* v_val_1064_; lean_object* v___x_1065_; lean_object* v___x_1066_; lean_object* v___x_1067_; lean_object* v___x_1068_; 
v_val_1064_ = lean_ctor_get(v___y_1059_, 0);
lean_inc(v_val_1064_);
lean_dec_ref_known(v___y_1059_, 1);
v___x_1065_ = l_Lean_SourceInfo_fromRef(v_val_1064_, v___x_982_);
lean_dec(v_val_1064_);
v___x_1066_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpIntro______elabRules__Mathlib__Tactic__tacticSimp__intro___________x2e_x2eOnly____1___closed__4));
v___x_1067_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1067_, 0, v___x_1065_);
lean_ctor_set(v___x_1067_, 1, v___x_1066_);
v___x_1068_ = l_Array_mkArray1___redArg(v___x_1067_);
v___y_1027_ = v___y_1051_;
v___y_1028_ = v___y_1052_;
v___y_1029_ = v___y_1053_;
v___y_1030_ = v___y_1054_;
v___y_1031_ = v___y_1052_;
v___y_1032_ = v___y_1055_;
v___y_1033_ = v___y_1056_;
v___y_1034_ = v___x_1063_;
v___y_1035_ = v___y_1057_;
v___y_1036_ = v___y_1058_;
v___y_1037_ = v___y_1060_;
v___y_1038_ = v___x_1068_;
goto v___jp_1026_;
}
else
{
lean_object* v___x_1069_; 
lean_dec(v___y_1059_);
v___x_1069_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpIntro______elabRules__Mathlib__Tactic__tacticSimp__intro___________x2e_x2eOnly____1___closed__3));
v___y_1027_ = v___y_1051_;
v___y_1028_ = v___y_1052_;
v___y_1029_ = v___y_1053_;
v___y_1030_ = v___y_1054_;
v___y_1031_ = v___y_1052_;
v___y_1032_ = v___y_1055_;
v___y_1033_ = v___y_1056_;
v___y_1034_ = v___x_1063_;
v___y_1035_ = v___y_1057_;
v___y_1036_ = v___y_1058_;
v___y_1037_ = v___y_1060_;
v___y_1038_ = v___x_1069_;
goto v___jp_1026_;
}
}
v___jp_1070_:
{
lean_object* v_ref_1078_; uint8_t v___x_1079_; lean_object* v___x_1080_; lean_object* v___x_1081_; lean_object* v___x_1082_; lean_object* v___x_1083_; lean_object* v___x_1084_; lean_object* v___x_1085_; 
v_ref_1078_ = lean_ctor_get(v_a_977_, 5);
v___x_1079_ = 0;
v___x_1080_ = l_Lean_SourceInfo_fromRef(v_ref_1078_, v___x_1079_);
v___x_1081_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpIntro______elabRules__Mathlib__Tactic__tacticSimp__intro___________x2e_x2eOnly____1___closed__5));
lean_inc_ref(v___y_1074_);
lean_inc_ref(v___y_1076_);
v___x_1082_ = l_Lean_Name_mkStr4(v___y_1076_, v___y_1074_, v___x_980_, v___x_1081_);
lean_inc(v___x_1080_);
v___x_1083_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1083_, 0, v___x_1080_);
lean_ctor_set(v___x_1083_, 1, v___x_1081_);
v___x_1084_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpIntro______elabRules__Mathlib__Tactic__tacticSimp__intro___________x2e_x2eOnly____1___closed__7));
v___x_1085_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpIntro______elabRules__Mathlib__Tactic__tacticSimp__intro___________x2e_x2eOnly____1___closed__8, &lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpIntro______elabRules__Mathlib__Tactic__tacticSimp__intro___________x2e_x2eOnly____1___closed__8_once, _init_lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpIntro______elabRules__Mathlib__Tactic__tacticSimp__intro___________x2e_x2eOnly____1___closed__8);
if (lean_obj_tag(v___y_1073_) == 0)
{
lean_object* v___x_1086_; 
v___x_1086_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpIntro______elabRules__Mathlib__Tactic__tacticSimp__intro___________x2e_x2eOnly____1___closed__3));
v___y_1051_ = v___y_1071_;
v___y_1052_ = v___x_1079_;
v___y_1053_ = v___y_1072_;
v___y_1054_ = v___x_1084_;
v___y_1055_ = v___x_1082_;
v___y_1056_ = v___x_1083_;
v___y_1057_ = v___x_1080_;
v___y_1058_ = v___x_1085_;
v___y_1059_ = v___y_1075_;
v___y_1060_ = v___y_1077_;
v___y_1061_ = v___x_1086_;
goto v___jp_1050_;
}
else
{
lean_object* v_val_1087_; lean_object* v___x_1088_; lean_object* v___x_1089_; 
v_val_1087_ = lean_ctor_get(v___y_1073_, 0);
lean_inc(v_val_1087_);
lean_dec_ref_known(v___y_1073_, 1);
v___x_1088_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpIntro______elabRules__Mathlib__Tactic__tacticSimp__intro___________x2e_x2eOnly____1___closed__3));
v___x_1089_ = lean_array_push(v___x_1088_, v_val_1087_);
v___y_1051_ = v___y_1071_;
v___y_1052_ = v___x_1079_;
v___y_1053_ = v___y_1072_;
v___y_1054_ = v___x_1084_;
v___y_1055_ = v___x_1082_;
v___y_1056_ = v___x_1083_;
v___y_1057_ = v___x_1080_;
v___y_1058_ = v___x_1085_;
v___y_1059_ = v___y_1075_;
v___y_1060_ = v___y_1077_;
v___y_1061_ = v___x_1089_;
goto v___jp_1050_;
}
}
v___jp_1090_:
{
lean_object* v___x_1097_; 
v___x_1097_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpIntro______elabRules__Mathlib__Tactic__tacticSimp__intro___________x2e_x2eOnly____1___closed__9));
if (lean_obj_tag(v___y_1093_) == 0)
{
lean_object* v___x_1098_; 
v___x_1098_ = lean_box(0);
v___y_1071_ = v___y_1092_;
v___y_1072_ = v___y_1094_;
v___y_1073_ = v___y_1096_;
v___y_1074_ = v___x_1097_;
v___y_1075_ = v___y_1091_;
v___y_1076_ = v___y_1095_;
v___y_1077_ = v___x_1098_;
goto v___jp_1070_;
}
else
{
lean_object* v_val_1099_; lean_object* v___x_1101_; uint8_t v_isShared_1102_; uint8_t v_isSharedCheck_1108_; 
v_val_1099_ = lean_ctor_get(v___y_1093_, 0);
v_isSharedCheck_1108_ = !lean_is_exclusive(v___y_1093_);
if (v_isSharedCheck_1108_ == 0)
{
v___x_1101_ = v___y_1093_;
v_isShared_1102_ = v_isSharedCheck_1108_;
goto v_resetjp_1100_;
}
else
{
lean_inc(v_val_1099_);
lean_dec(v___y_1093_);
v___x_1101_ = lean_box(0);
v_isShared_1102_ = v_isSharedCheck_1108_;
goto v_resetjp_1100_;
}
v_resetjp_1100_:
{
lean_object* v___x_1103_; lean_object* v___x_1104_; lean_object* v___x_1106_; 
v___x_1103_ = l_Lean_Syntax_getArg(v_val_1099_, v___x_984_);
lean_dec(v_val_1099_);
v___x_1104_ = l_Lean_Syntax_getArgs(v___x_1103_);
lean_dec(v___x_1103_);
if (v_isShared_1102_ == 0)
{
lean_ctor_set(v___x_1101_, 0, v___x_1104_);
v___x_1106_ = v___x_1101_;
goto v_reusejp_1105_;
}
else
{
lean_object* v_reuseFailAlloc_1107_; 
v_reuseFailAlloc_1107_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1107_, 0, v___x_1104_);
v___x_1106_ = v_reuseFailAlloc_1107_;
goto v_reusejp_1105_;
}
v_reusejp_1105_:
{
v___y_1071_ = v___y_1092_;
v___y_1072_ = v___y_1094_;
v___y_1073_ = v___y_1096_;
v___y_1074_ = v___x_1097_;
v___y_1075_ = v___y_1091_;
v___y_1076_ = v___y_1095_;
v___y_1077_ = v___x_1106_;
goto v___jp_1070_;
}
}
}
}
v___jp_1113_:
{
lean_object* v___x_1117_; lean_object* v_ids_1118_; lean_object* v___x_1119_; 
v___x_1117_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpIntro______elabRules__Mathlib__Tactic__tacticSimp__intro___________x2e_x2eOnly____1___closed__10));
v_ids_1118_ = l_Lean_Syntax_getArgs(v___x_1112_);
lean_dec(v___x_1112_);
v___x_1119_ = l_Lean_Syntax_getOptional_x3f(v___x_1110_);
lean_dec(v___x_1110_);
if (lean_obj_tag(v___x_1119_) == 0)
{
lean_object* v___x_1120_; 
v___x_1120_ = lean_box(0);
v___y_1091_ = v___y_1114_;
v___y_1092_ = v___y_1116_;
v___y_1093_ = v___y_1115_;
v___y_1094_ = v_ids_1118_;
v___y_1095_ = v___x_1117_;
v___y_1096_ = v___x_1120_;
goto v___jp_1090_;
}
else
{
lean_object* v_val_1121_; lean_object* v___x_1123_; uint8_t v_isShared_1124_; uint8_t v_isSharedCheck_1128_; 
v_val_1121_ = lean_ctor_get(v___x_1119_, 0);
v_isSharedCheck_1128_ = !lean_is_exclusive(v___x_1119_);
if (v_isSharedCheck_1128_ == 0)
{
v___x_1123_ = v___x_1119_;
v_isShared_1124_ = v_isSharedCheck_1128_;
goto v_resetjp_1122_;
}
else
{
lean_inc(v_val_1121_);
lean_dec(v___x_1119_);
v___x_1123_ = lean_box(0);
v_isShared_1124_ = v_isSharedCheck_1128_;
goto v_resetjp_1122_;
}
v_resetjp_1122_:
{
lean_object* v___x_1126_; 
if (v_isShared_1124_ == 0)
{
v___x_1126_ = v___x_1123_;
goto v_reusejp_1125_;
}
else
{
lean_object* v_reuseFailAlloc_1127_; 
v_reuseFailAlloc_1127_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1127_, 0, v_val_1121_);
v___x_1126_ = v_reuseFailAlloc_1127_;
goto v_reusejp_1125_;
}
v_reusejp_1125_:
{
v___y_1091_ = v___y_1114_;
v___y_1092_ = v___y_1116_;
v___y_1093_ = v___y_1115_;
v___y_1094_ = v_ids_1118_;
v___y_1095_ = v___x_1117_;
v___y_1096_ = v___x_1126_;
goto v___jp_1090_;
}
}
}
}
v___jp_1131_:
{
lean_object* v___x_1134_; 
v___x_1134_ = l_Lean_Syntax_getOptional_x3f(v___x_1130_);
lean_dec(v___x_1130_);
if (lean_obj_tag(v___x_1134_) == 0)
{
lean_object* v___x_1135_; 
v___x_1135_ = lean_box(0);
v___y_1114_ = v___y_1133_;
v___y_1115_ = v___y_1132_;
v___y_1116_ = v___x_1135_;
goto v___jp_1113_;
}
else
{
lean_object* v_val_1136_; lean_object* v___x_1138_; uint8_t v_isShared_1139_; uint8_t v_isSharedCheck_1143_; 
v_val_1136_ = lean_ctor_get(v___x_1134_, 0);
v_isSharedCheck_1143_ = !lean_is_exclusive(v___x_1134_);
if (v_isSharedCheck_1143_ == 0)
{
v___x_1138_ = v___x_1134_;
v_isShared_1139_ = v_isSharedCheck_1143_;
goto v_resetjp_1137_;
}
else
{
lean_inc(v_val_1136_);
lean_dec(v___x_1134_);
v___x_1138_ = lean_box(0);
v_isShared_1139_ = v_isSharedCheck_1143_;
goto v_resetjp_1137_;
}
v_resetjp_1137_:
{
lean_object* v___x_1141_; 
if (v_isShared_1139_ == 0)
{
v___x_1141_ = v___x_1138_;
goto v_reusejp_1140_;
}
else
{
lean_object* v_reuseFailAlloc_1142_; 
v_reuseFailAlloc_1142_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1142_, 0, v_val_1136_);
v___x_1141_ = v_reuseFailAlloc_1142_;
goto v_reusejp_1140_;
}
v_reusejp_1140_:
{
v___y_1114_ = v___y_1133_;
v___y_1115_ = v___y_1132_;
v___y_1116_ = v___x_1141_;
goto v___jp_1113_;
}
}
}
}
v___jp_1146_:
{
lean_object* v___x_1148_; 
v___x_1148_ = l_Lean_Syntax_getOptional_x3f(v___x_1145_);
lean_dec(v___x_1145_);
if (lean_obj_tag(v___x_1148_) == 0)
{
lean_object* v___x_1149_; 
v___x_1149_ = lean_box(0);
v___y_1132_ = v___y_1147_;
v___y_1133_ = v___x_1149_;
goto v___jp_1131_;
}
else
{
lean_object* v_val_1150_; lean_object* v___x_1152_; uint8_t v_isShared_1153_; uint8_t v_isSharedCheck_1157_; 
v_val_1150_ = lean_ctor_get(v___x_1148_, 0);
v_isSharedCheck_1157_ = !lean_is_exclusive(v___x_1148_);
if (v_isSharedCheck_1157_ == 0)
{
v___x_1152_ = v___x_1148_;
v_isShared_1153_ = v_isSharedCheck_1157_;
goto v_resetjp_1151_;
}
else
{
lean_inc(v_val_1150_);
lean_dec(v___x_1148_);
v___x_1152_ = lean_box(0);
v_isShared_1153_ = v_isSharedCheck_1157_;
goto v_resetjp_1151_;
}
v_resetjp_1151_:
{
lean_object* v___x_1155_; 
if (v_isShared_1153_ == 0)
{
v___x_1155_ = v___x_1152_;
goto v_reusejp_1154_;
}
else
{
lean_object* v_reuseFailAlloc_1156_; 
v_reuseFailAlloc_1156_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1156_, 0, v_val_1150_);
v___x_1155_ = v_reuseFailAlloc_1156_;
goto v_reusejp_1154_;
}
v_reusejp_1154_:
{
v___y_1132_ = v___y_1147_;
v___y_1133_ = v___x_1155_;
goto v___jp_1131_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpIntro______elabRules__Mathlib__Tactic__tacticSimp__intro___________x2e_x2eOnly____1___boxed(lean_object* v_x_1170_, lean_object* v_a_1171_, lean_object* v_a_1172_, lean_object* v_a_1173_, lean_object* v_a_1174_, lean_object* v_a_1175_, lean_object* v_a_1176_, lean_object* v_a_1177_, lean_object* v_a_1178_, lean_object* v_a_1179_){
_start:
{
lean_object* v_res_1180_; 
v_res_1180_ = lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__SimpIntro______elabRules__Mathlib__Tactic__tacticSimp__intro___________x2e_x2eOnly____1(v_x_1170_, v_a_1171_, v_a_1172_, v_a_1173_, v_a_1174_, v_a_1175_, v_a_1176_, v_a_1177_, v_a_1178_);
lean_dec(v_a_1178_);
lean_dec_ref(v_a_1177_);
lean_dec(v_a_1176_);
lean_dec_ref(v_a_1175_);
lean_dec(v_a_1174_);
lean_dec_ref(v_a_1173_);
lean_dec(v_a_1172_);
lean_dec_ref(v_a_1171_);
return v_res_1180_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Init(uint8_t builtin);
lean_object* runtime_initialize_Lean_Elab_Tactic_Simp(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Tactic_SimpIntro(uint8_t builtin) {
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
res = runtime_initialize_Lean_Elab_Tactic_Simp(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_Lean_Elab_Tactic_Simp(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Tactic_SimpIntro(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Elab_Tactic_Simp(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly__ = _init_lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly__();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_tacticSimp__intro___________x2e_x2eOnly__);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Lean_Elab_Tactic_Simp(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Init(uint8_t builtin);
lean_object* initialize_Lean_Elab_Tactic_Simp(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_SimpIntro(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Elab_Tactic_Simp(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Elab_Tactic_Simp(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_SimpIntro(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Tactic_SimpIntro(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Tactic_SimpIntro(builtin);
}
#ifdef __cplusplus
}
#endif
