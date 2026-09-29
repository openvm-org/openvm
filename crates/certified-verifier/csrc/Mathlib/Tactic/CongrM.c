// Lean compiler output
// Module: Mathlib.Tactic.CongrM
// Imports: public import Init public meta import Init public import Mathlib.Tactic.Relation.Rfl public import Mathlib.Tactic.TermCongr
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
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Lean_Name_str___override(lean_object*, lean_object*);
lean_object* l_Lean_Name_num___override(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* lean_st_ref_get(lean_object*);
lean_object* lean_st_ref_take(lean_object*);
double lean_float_of_nat(lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* l_Lean_PersistentArray_push___redArg(lean_object*, lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_registerTraceClass(lean_object*, uint8_t, lean_object*);
lean_object* l_Lean_Elab_Tactic_getMainGoal___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MVarId_iffOfEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Tactic_liftReflToEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_replaceMainGoal___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_String_toRawSubstring_x27(lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isAntiquots(lean_object*);
lean_object* l_Lean_Syntax_mkAntiquotNode(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t);
size_t lean_array_size(lean_object*);
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
size_t lean_usize_add(size_t, size_t);
lean_object* l_Lean_Name_mkStr3(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_getMainTarget(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Term_exprToSyntax(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node1(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_evalTactic(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_withMainContext___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_append(lean_object*, lean_object*);
uint8_t l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* l_Lean_MessageData_ofSyntax(lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_CongrM_0__Mathlib_Tactic_initFn___closed__0_00___x40_Mathlib_Tactic_CongrM_1572680169____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_CongrM_0__Mathlib_Tactic_initFn___closed__0_00___x40_Mathlib_Tactic_CongrM_1572680169____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_CongrM_0__Mathlib_Tactic_initFn___closed__0_00___x40_Mathlib_Tactic_CongrM_1572680169____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_CongrM_0__Mathlib_Tactic_initFn___closed__1_00___x40_Mathlib_Tactic_CongrM_1572680169____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "congrm"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_CongrM_0__Mathlib_Tactic_initFn___closed__1_00___x40_Mathlib_Tactic_CongrM_1572680169____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_CongrM_0__Mathlib_Tactic_initFn___closed__1_00___x40_Mathlib_Tactic_CongrM_1572680169____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_CongrM_0__Mathlib_Tactic_initFn___closed__2_00___x40_Mathlib_Tactic_CongrM_1572680169____hygCtx___hyg_2__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CongrM_0__Mathlib_Tactic_initFn___closed__0_00___x40_Mathlib_Tactic_CongrM_1572680169____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(186, 205, 46, 93, 234, 75, 44, 75)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_CongrM_0__Mathlib_Tactic_initFn___closed__2_00___x40_Mathlib_Tactic_CongrM_1572680169____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CongrM_0__Mathlib_Tactic_initFn___closed__2_00___x40_Mathlib_Tactic_CongrM_1572680169____hygCtx___hyg_2__value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CongrM_0__Mathlib_Tactic_initFn___closed__1_00___x40_Mathlib_Tactic_CongrM_1572680169____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(131, 184, 87, 39, 107, 220, 132, 220)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_CongrM_0__Mathlib_Tactic_initFn___closed__2_00___x40_Mathlib_Tactic_CongrM_1572680169____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_CongrM_0__Mathlib_Tactic_initFn___closed__2_00___x40_Mathlib_Tactic_CongrM_1572680169____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_CongrM_0__Mathlib_Tactic_initFn___closed__3_00___x40_Mathlib_Tactic_CongrM_1572680169____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "_private"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_CongrM_0__Mathlib_Tactic_initFn___closed__3_00___x40_Mathlib_Tactic_CongrM_1572680169____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_CongrM_0__Mathlib_Tactic_initFn___closed__3_00___x40_Mathlib_Tactic_CongrM_1572680169____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_CongrM_0__Mathlib_Tactic_initFn___closed__4_00___x40_Mathlib_Tactic_CongrM_1572680169____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CongrM_0__Mathlib_Tactic_initFn___closed__3_00___x40_Mathlib_Tactic_CongrM_1572680169____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(103, 214, 75, 80, 34, 198, 193, 153)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_CongrM_0__Mathlib_Tactic_initFn___closed__4_00___x40_Mathlib_Tactic_CongrM_1572680169____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_CongrM_0__Mathlib_Tactic_initFn___closed__4_00___x40_Mathlib_Tactic_CongrM_1572680169____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_CongrM_0__Mathlib_Tactic_initFn___closed__5_00___x40_Mathlib_Tactic_CongrM_1572680169____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_CongrM_0__Mathlib_Tactic_initFn___closed__5_00___x40_Mathlib_Tactic_CongrM_1572680169____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_CongrM_0__Mathlib_Tactic_initFn___closed__5_00___x40_Mathlib_Tactic_CongrM_1572680169____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_CongrM_0__Mathlib_Tactic_initFn___closed__6_00___x40_Mathlib_Tactic_CongrM_1572680169____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CongrM_0__Mathlib_Tactic_initFn___closed__4_00___x40_Mathlib_Tactic_CongrM_1572680169____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CongrM_0__Mathlib_Tactic_initFn___closed__5_00___x40_Mathlib_Tactic_CongrM_1572680169____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(234, 232, 174, 134, 127, 136, 69, 92)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_CongrM_0__Mathlib_Tactic_initFn___closed__6_00___x40_Mathlib_Tactic_CongrM_1572680169____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_CongrM_0__Mathlib_Tactic_initFn___closed__6_00___x40_Mathlib_Tactic_CongrM_1572680169____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_CongrM_0__Mathlib_Tactic_initFn___closed__7_00___x40_Mathlib_Tactic_CongrM_1572680169____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CongrM_0__Mathlib_Tactic_initFn___closed__6_00___x40_Mathlib_Tactic_CongrM_1572680169____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CongrM_0__Mathlib_Tactic_initFn___closed__0_00___x40_Mathlib_Tactic_CongrM_1572680169____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(191, 70, 156, 159, 11, 54, 216, 94)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_CongrM_0__Mathlib_Tactic_initFn___closed__7_00___x40_Mathlib_Tactic_CongrM_1572680169____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_CongrM_0__Mathlib_Tactic_initFn___closed__7_00___x40_Mathlib_Tactic_CongrM_1572680169____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_CongrM_0__Mathlib_Tactic_initFn___closed__8_00___x40_Mathlib_Tactic_CongrM_1572680169____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "CongrM"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_CongrM_0__Mathlib_Tactic_initFn___closed__8_00___x40_Mathlib_Tactic_CongrM_1572680169____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_CongrM_0__Mathlib_Tactic_initFn___closed__8_00___x40_Mathlib_Tactic_CongrM_1572680169____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_CongrM_0__Mathlib_Tactic_initFn___closed__9_00___x40_Mathlib_Tactic_CongrM_1572680169____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CongrM_0__Mathlib_Tactic_initFn___closed__7_00___x40_Mathlib_Tactic_CongrM_1572680169____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CongrM_0__Mathlib_Tactic_initFn___closed__8_00___x40_Mathlib_Tactic_CongrM_1572680169____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(59, 212, 13, 161, 57, 60, 4, 151)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_CongrM_0__Mathlib_Tactic_initFn___closed__9_00___x40_Mathlib_Tactic_CongrM_1572680169____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_CongrM_0__Mathlib_Tactic_initFn___closed__9_00___x40_Mathlib_Tactic_CongrM_1572680169____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_CongrM_0__Mathlib_Tactic_initFn___closed__10_00___x40_Mathlib_Tactic_CongrM_1572680169____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CongrM_0__Mathlib_Tactic_initFn___closed__9_00___x40_Mathlib_Tactic_CongrM_1572680169____hygCtx___hyg_2__value),((lean_object*)(((size_t)(0) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(70, 68, 97, 171, 123, 61, 24, 94)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_CongrM_0__Mathlib_Tactic_initFn___closed__10_00___x40_Mathlib_Tactic_CongrM_1572680169____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_CongrM_0__Mathlib_Tactic_initFn___closed__10_00___x40_Mathlib_Tactic_CongrM_1572680169____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_CongrM_0__Mathlib_Tactic_initFn___closed__11_00___x40_Mathlib_Tactic_CongrM_1572680169____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CongrM_0__Mathlib_Tactic_initFn___closed__10_00___x40_Mathlib_Tactic_CongrM_1572680169____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CongrM_0__Mathlib_Tactic_initFn___closed__5_00___x40_Mathlib_Tactic_CongrM_1572680169____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(31, 171, 135, 30, 8, 126, 180, 168)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_CongrM_0__Mathlib_Tactic_initFn___closed__11_00___x40_Mathlib_Tactic_CongrM_1572680169____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_CongrM_0__Mathlib_Tactic_initFn___closed__11_00___x40_Mathlib_Tactic_CongrM_1572680169____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_CongrM_0__Mathlib_Tactic_initFn___closed__12_00___x40_Mathlib_Tactic_CongrM_1572680169____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CongrM_0__Mathlib_Tactic_initFn___closed__11_00___x40_Mathlib_Tactic_CongrM_1572680169____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CongrM_0__Mathlib_Tactic_initFn___closed__0_00___x40_Mathlib_Tactic_CongrM_1572680169____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(46, 42, 177, 58, 4, 6, 168, 237)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_CongrM_0__Mathlib_Tactic_initFn___closed__12_00___x40_Mathlib_Tactic_CongrM_1572680169____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_CongrM_0__Mathlib_Tactic_initFn___closed__12_00___x40_Mathlib_Tactic_CongrM_1572680169____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_CongrM_0__Mathlib_Tactic_initFn___closed__13_00___x40_Mathlib_Tactic_CongrM_1572680169____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "initFn"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_CongrM_0__Mathlib_Tactic_initFn___closed__13_00___x40_Mathlib_Tactic_CongrM_1572680169____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_CongrM_0__Mathlib_Tactic_initFn___closed__13_00___x40_Mathlib_Tactic_CongrM_1572680169____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_CongrM_0__Mathlib_Tactic_initFn___closed__14_00___x40_Mathlib_Tactic_CongrM_1572680169____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CongrM_0__Mathlib_Tactic_initFn___closed__12_00___x40_Mathlib_Tactic_CongrM_1572680169____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CongrM_0__Mathlib_Tactic_initFn___closed__13_00___x40_Mathlib_Tactic_CongrM_1572680169____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(67, 188, 168, 120, 82, 250, 252, 46)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_CongrM_0__Mathlib_Tactic_initFn___closed__14_00___x40_Mathlib_Tactic_CongrM_1572680169____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_CongrM_0__Mathlib_Tactic_initFn___closed__14_00___x40_Mathlib_Tactic_CongrM_1572680169____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_CongrM_0__Mathlib_Tactic_initFn___closed__15_00___x40_Mathlib_Tactic_CongrM_1572680169____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "_@"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_CongrM_0__Mathlib_Tactic_initFn___closed__15_00___x40_Mathlib_Tactic_CongrM_1572680169____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_CongrM_0__Mathlib_Tactic_initFn___closed__15_00___x40_Mathlib_Tactic_CongrM_1572680169____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_CongrM_0__Mathlib_Tactic_initFn___closed__16_00___x40_Mathlib_Tactic_CongrM_1572680169____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CongrM_0__Mathlib_Tactic_initFn___closed__14_00___x40_Mathlib_Tactic_CongrM_1572680169____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CongrM_0__Mathlib_Tactic_initFn___closed__15_00___x40_Mathlib_Tactic_CongrM_1572680169____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(182, 93, 187, 250, 101, 118, 87, 86)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_CongrM_0__Mathlib_Tactic_initFn___closed__16_00___x40_Mathlib_Tactic_CongrM_1572680169____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_CongrM_0__Mathlib_Tactic_initFn___closed__16_00___x40_Mathlib_Tactic_CongrM_1572680169____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_CongrM_0__Mathlib_Tactic_initFn___closed__17_00___x40_Mathlib_Tactic_CongrM_1572680169____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CongrM_0__Mathlib_Tactic_initFn___closed__16_00___x40_Mathlib_Tactic_CongrM_1572680169____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CongrM_0__Mathlib_Tactic_initFn___closed__5_00___x40_Mathlib_Tactic_CongrM_1572680169____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(15, 127, 223, 32, 159, 85, 138, 18)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_CongrM_0__Mathlib_Tactic_initFn___closed__17_00___x40_Mathlib_Tactic_CongrM_1572680169____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_CongrM_0__Mathlib_Tactic_initFn___closed__17_00___x40_Mathlib_Tactic_CongrM_1572680169____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_CongrM_0__Mathlib_Tactic_initFn___closed__18_00___x40_Mathlib_Tactic_CongrM_1572680169____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CongrM_0__Mathlib_Tactic_initFn___closed__17_00___x40_Mathlib_Tactic_CongrM_1572680169____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CongrM_0__Mathlib_Tactic_initFn___closed__0_00___x40_Mathlib_Tactic_CongrM_1572680169____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(222, 244, 36, 96, 52, 159, 129, 154)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_CongrM_0__Mathlib_Tactic_initFn___closed__18_00___x40_Mathlib_Tactic_CongrM_1572680169____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_CongrM_0__Mathlib_Tactic_initFn___closed__18_00___x40_Mathlib_Tactic_CongrM_1572680169____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_CongrM_0__Mathlib_Tactic_initFn___closed__19_00___x40_Mathlib_Tactic_CongrM_1572680169____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CongrM_0__Mathlib_Tactic_initFn___closed__18_00___x40_Mathlib_Tactic_CongrM_1572680169____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CongrM_0__Mathlib_Tactic_initFn___closed__8_00___x40_Mathlib_Tactic_CongrM_1572680169____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(134, 232, 166, 61, 141, 68, 199, 22)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_CongrM_0__Mathlib_Tactic_initFn___closed__19_00___x40_Mathlib_Tactic_CongrM_1572680169____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_CongrM_0__Mathlib_Tactic_initFn___closed__19_00___x40_Mathlib_Tactic_CongrM_1572680169____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_CongrM_0__Mathlib_Tactic_initFn___closed__20_00___x40_Mathlib_Tactic_CongrM_1572680169____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CongrM_0__Mathlib_Tactic_initFn___closed__19_00___x40_Mathlib_Tactic_CongrM_1572680169____hygCtx___hyg_2__value),((lean_object*)(((size_t)(1572680169) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(9, 121, 164, 163, 71, 167, 219, 61)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_CongrM_0__Mathlib_Tactic_initFn___closed__20_00___x40_Mathlib_Tactic_CongrM_1572680169____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_CongrM_0__Mathlib_Tactic_initFn___closed__20_00___x40_Mathlib_Tactic_CongrM_1572680169____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_CongrM_0__Mathlib_Tactic_initFn___closed__21_00___x40_Mathlib_Tactic_CongrM_1572680169____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "_hygCtx"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_CongrM_0__Mathlib_Tactic_initFn___closed__21_00___x40_Mathlib_Tactic_CongrM_1572680169____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_CongrM_0__Mathlib_Tactic_initFn___closed__21_00___x40_Mathlib_Tactic_CongrM_1572680169____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_CongrM_0__Mathlib_Tactic_initFn___closed__22_00___x40_Mathlib_Tactic_CongrM_1572680169____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CongrM_0__Mathlib_Tactic_initFn___closed__20_00___x40_Mathlib_Tactic_CongrM_1572680169____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CongrM_0__Mathlib_Tactic_initFn___closed__21_00___x40_Mathlib_Tactic_CongrM_1572680169____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(154, 204, 216, 54, 86, 52, 185, 240)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_CongrM_0__Mathlib_Tactic_initFn___closed__22_00___x40_Mathlib_Tactic_CongrM_1572680169____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_CongrM_0__Mathlib_Tactic_initFn___closed__22_00___x40_Mathlib_Tactic_CongrM_1572680169____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_CongrM_0__Mathlib_Tactic_initFn___closed__23_00___x40_Mathlib_Tactic_CongrM_1572680169____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "_hyg"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_CongrM_0__Mathlib_Tactic_initFn___closed__23_00___x40_Mathlib_Tactic_CongrM_1572680169____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_CongrM_0__Mathlib_Tactic_initFn___closed__23_00___x40_Mathlib_Tactic_CongrM_1572680169____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_CongrM_0__Mathlib_Tactic_initFn___closed__24_00___x40_Mathlib_Tactic_CongrM_1572680169____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CongrM_0__Mathlib_Tactic_initFn___closed__22_00___x40_Mathlib_Tactic_CongrM_1572680169____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CongrM_0__Mathlib_Tactic_initFn___closed__23_00___x40_Mathlib_Tactic_CongrM_1572680169____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(230, 198, 152, 78, 219, 20, 5, 23)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_CongrM_0__Mathlib_Tactic_initFn___closed__24_00___x40_Mathlib_Tactic_CongrM_1572680169____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_CongrM_0__Mathlib_Tactic_initFn___closed__24_00___x40_Mathlib_Tactic_CongrM_1572680169____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_CongrM_0__Mathlib_Tactic_initFn___closed__25_00___x40_Mathlib_Tactic_CongrM_1572680169____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CongrM_0__Mathlib_Tactic_initFn___closed__24_00___x40_Mathlib_Tactic_CongrM_1572680169____hygCtx___hyg_2__value),((lean_object*)(((size_t)(2) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(39, 23, 23, 29, 224, 111, 168, 216)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_CongrM_0__Mathlib_Tactic_initFn___closed__25_00___x40_Mathlib_Tactic_CongrM_1572680169____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_CongrM_0__Mathlib_Tactic_initFn___closed__25_00___x40_Mathlib_Tactic_CongrM_1572680169____hygCtx___hyg_2__value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_CongrM_0__Mathlib_Tactic_initFn_00___x40_Mathlib_Tactic_CongrM_1572680169____hygCtx___hyg_2_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_CongrM_0__Mathlib_Tactic_initFn_00___x40_Mathlib_Tactic_CongrM_1572680169____hygCtx___hyg_2____boxed(lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_congrM___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "congrM"};
static const lean_object* lp_mathlib_Mathlib_Tactic_congrM___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_congrM___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_congrM___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CongrM_0__Mathlib_Tactic_initFn___closed__5_00___x40_Mathlib_Tactic_CongrM_1572680169____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_congrM___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_congrM___closed__1_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CongrM_0__Mathlib_Tactic_initFn___closed__0_00___x40_Mathlib_Tactic_CongrM_1572680169____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_congrM___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_congrM___closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_congrM___closed__0_value),LEAN_SCALAR_PTR_LITERAL(248, 64, 190, 53, 85, 54, 115, 66)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_congrM___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_congrM___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_congrM___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_Mathlib_Tactic_congrM___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_congrM___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_congrM___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_congrM___closed__2_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_congrM___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_congrM___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_congrM___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "congrm "};
static const lean_object* lp_mathlib_Mathlib_Tactic_congrM___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_congrM___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_congrM___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_congrM___closed__4_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_congrM___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_congrM___closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_congrM___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "term"};
static const lean_object* lp_mathlib_Mathlib_Tactic_congrM___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_congrM___closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_congrM___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_congrM___closed__6_value),LEAN_SCALAR_PTR_LITERAL(187, 230, 181, 162, 253, 146, 122, 119)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_congrM___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_congrM___closed__7_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_congrM___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_congrM___closed__7_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_congrM___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_congrM___closed__8_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_congrM___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_congrM___closed__3_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_congrM___closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_congrM___closed__8_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_congrM___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_congrM___closed__9_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_congrM___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_congrM___closed__1_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_congrM___closed__9_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_congrM___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_congrM___closed__10_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Tactic_congrM = (const lean_object*)&lp_mathlib_Mathlib_Tactic_congrM___closed__10_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1_spec__0___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1_spec__0___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1_spec__0___redArg();
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1_spec__0___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "refine"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "typeAscription"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___closed__5_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___closed__5_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___closed__5_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___closed__5_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___closed__5_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___closed__5_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___closed__4_value),LEAN_SCALAR_PTR_LITERAL(247, 209, 88, 141, 5, 195, 49, 74)}};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "hygienicLParen"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___closed__7_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___closed__7_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___closed__7_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___closed__7_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___closed__7_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___closed__7_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___closed__6_value),LEAN_SCALAR_PTR_LITERAL(41, 104, 206, 51, 21, 254, 100, 101)}};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___closed__7_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "("};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___closed__8_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "hygieneInfo"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___closed__9_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___closed__9_value),LEAN_SCALAR_PTR_LITERAL(27, 64, 36, 144, 170, 151, 255, 136)}};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___closed__10_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1, .m_capacity = 1, .m_length = 0, .m_data = ""};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___closed__11_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___closed__12_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___closed__12;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Meta"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___closed__13_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___closed__14_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___closed__14_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___closed__13_value),LEAN_SCALAR_PTR_LITERAL(194, 50, 106, 158, 41, 60, 103, 214)}};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___closed__14 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___closed__14_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___closed__14_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___closed__15 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___closed__15_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Elab"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___closed__16 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___closed__16_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___closed__17_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___closed__17_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___closed__16_value),LEAN_SCALAR_PTR_LITERAL(52, 247, 248, 201, 92, 23, 188, 159)}};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___closed__17 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___closed__17_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___closed__17_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___closed__18 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___closed__18_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___closed__19_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___closed__19_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___closed__19 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___closed__19_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___closed__19_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___closed__20 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___closed__20_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___closed__21 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___closed__21_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___closed__21_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___closed__22 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___closed__22_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___closed__22_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___closed__23 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___closed__23_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___closed__20_value),((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___closed__23_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___closed__24 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___closed__24_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___closed__18_value),((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___closed__24_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___closed__25 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___closed__25_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___closed__26_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "TermCongr"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___closed__26 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___closed__26_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___closed__27_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "termCongr"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___closed__27 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___closed__27_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___closed__28_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "congr("};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___closed__28 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___closed__28_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___closed__29_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ")"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___closed__29 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___closed__29_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___closed__30_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ":"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___closed__30 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___closed__30_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___closed__31_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___closed__31 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___closed__31_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___closed__32_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___closed__31_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___closed__32 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___closed__32_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_addTrace___at___00Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1_spec__2_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_addTrace___at___00Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1_spec__2_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1_spec__2___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static double lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1_spec__2___redArg___closed__0;
static const lean_array_object lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1_spec__2___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1_spec__2___redArg___closed__1 = (const lean_object*)&lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1_spec__2___redArg___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1_spec__2___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_Syntax_replaceM___at___00Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1_spec__1___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "syntheticHole"};
static const lean_object* lp_mathlib_Lean_Syntax_replaceM___at___00Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1_spec__1___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Lean_Syntax_replaceM___at___00Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1_spec__1___lam__0___closed__0_value;
static const lean_ctor_object lp_mathlib_Lean_Syntax_replaceM___at___00Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1_spec__1___lam__0___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Lean_Syntax_replaceM___at___00Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1_spec__1___lam__0___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Syntax_replaceM___at___00Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1_spec__1___lam__0___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Lean_Syntax_replaceM___at___00Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1_spec__1___lam__0___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Syntax_replaceM___at___00Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1_spec__1___lam__0___closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Lean_Syntax_replaceM___at___00Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1_spec__1___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Syntax_replaceM___at___00Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1_spec__1___lam__0___closed__1_value_aux_2),((lean_object*)&lp_mathlib_Lean_Syntax_replaceM___at___00Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1_spec__1___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(218, 189, 67, 60, 211, 196, 112, 165)}};
static const lean_object* lp_mathlib_Lean_Syntax_replaceM___at___00Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1_spec__1___lam__0___closed__1 = (const lean_object*)&lp_mathlib_Lean_Syntax_replaceM___at___00Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1_spec__1___lam__0___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Syntax_replaceM___at___00Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1_spec__1___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Syntax_replaceM___at___00Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1_spec__1___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Syntax_replaceM___at___00Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Syntax_replaceM___at___00Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1_spec__1_spec__1(size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Syntax_replaceM___at___00Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1_spec__1_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Syntax_replaceM___at___00Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__0___boxed, .m_arity = 9, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "trace"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(212, 145, 141, 177, 67, 149, 127, 197)}};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___closed__2_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___closed__3;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "pattern: "};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___closed__4_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___closed__5;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_CongrM_0__Mathlib_Tactic_initFn_00___x40_Mathlib_Tactic_CongrM_1572680169____hygCtx___hyg_2_(){
_start:
{
lean_object* v___x_62_; uint8_t v___x_63_; lean_object* v___x_64_; lean_object* v___x_65_; 
v___x_62_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_CongrM_0__Mathlib_Tactic_initFn___closed__2_00___x40_Mathlib_Tactic_CongrM_1572680169____hygCtx___hyg_2_));
v___x_63_ = 0;
v___x_64_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_CongrM_0__Mathlib_Tactic_initFn___closed__25_00___x40_Mathlib_Tactic_CongrM_1572680169____hygCtx___hyg_2_));
v___x_65_ = l_Lean_registerTraceClass(v___x_62_, v___x_63_, v___x_64_);
return v___x_65_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_CongrM_0__Mathlib_Tactic_initFn_00___x40_Mathlib_Tactic_CongrM_1572680169____hygCtx___hyg_2____boxed(lean_object* v_a_66_){
_start:
{
lean_object* v_res_67_; 
v_res_67_ = lp_mathlib___private_Mathlib_Tactic_CongrM_0__Mathlib_Tactic_initFn_00___x40_Mathlib_Tactic_CongrM_1572680169____hygCtx___hyg_2_();
return v_res_67_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1_spec__0___redArg___closed__0(void){
_start:
{
lean_object* v___x_95_; lean_object* v___x_96_; lean_object* v___x_97_; 
v___x_95_ = lean_box(0);
v___x_96_ = l_Lean_Elab_unsupportedSyntaxExceptionId;
v___x_97_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_97_, 0, v___x_96_);
lean_ctor_set(v___x_97_, 1, v___x_95_);
return v___x_97_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1_spec__0___redArg(){
_start:
{
lean_object* v___x_99_; lean_object* v___x_100_; 
v___x_99_ = lean_obj_once(&lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1_spec__0___redArg___closed__0, &lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1_spec__0___redArg___closed__0_once, _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1_spec__0___redArg___closed__0);
v___x_100_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_100_, 0, v___x_99_);
return v___x_100_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1_spec__0___redArg___boxed(lean_object* v___y_101_){
_start:
{
lean_object* v_res_102_; 
v_res_102_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1_spec__0___redArg();
return v_res_102_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1_spec__0(lean_object* v_00_u03b1_103_, lean_object* v___y_104_, lean_object* v___y_105_, lean_object* v___y_106_, lean_object* v___y_107_, lean_object* v___y_108_, lean_object* v___y_109_, lean_object* v___y_110_, lean_object* v___y_111_){
_start:
{
lean_object* v___x_113_; 
v___x_113_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1_spec__0___redArg();
return v___x_113_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1_spec__0___boxed(lean_object* v_00_u03b1_114_, lean_object* v___y_115_, lean_object* v___y_116_, lean_object* v___y_117_, lean_object* v___y_118_, lean_object* v___y_119_, lean_object* v___y_120_, lean_object* v___y_121_, lean_object* v___y_122_, lean_object* v___y_123_){
_start:
{
lean_object* v_res_124_; 
v_res_124_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1_spec__0(v_00_u03b1_114_, v___y_115_, v___y_116_, v___y_117_, v___y_118_, v___y_119_, v___y_120_, v___y_121_, v___y_122_);
lean_dec(v___y_122_);
lean_dec_ref(v___y_121_);
lean_dec(v___y_120_);
lean_dec_ref(v___y_119_);
lean_dec(v___y_118_);
lean_dec_ref(v___y_117_);
lean_dec(v___y_116_);
lean_dec_ref(v___y_115_);
return v_res_124_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__0(lean_object* v___y_125_, lean_object* v___y_126_, lean_object* v___y_127_, lean_object* v___y_128_, lean_object* v___y_129_, lean_object* v___y_130_, lean_object* v___y_131_, lean_object* v___y_132_){
_start:
{
lean_object* v___x_134_; 
v___x_134_ = l_Lean_Elab_Tactic_getMainGoal___redArg(v___y_126_, v___y_129_, v___y_130_, v___y_131_, v___y_132_);
if (lean_obj_tag(v___x_134_) == 0)
{
lean_object* v_a_135_; lean_object* v___x_136_; 
v_a_135_ = lean_ctor_get(v___x_134_, 0);
lean_inc(v_a_135_);
lean_dec_ref_known(v___x_134_, 1);
v___x_136_ = l_Lean_MVarId_iffOfEq(v_a_135_, v___y_129_, v___y_130_, v___y_131_, v___y_132_);
if (lean_obj_tag(v___x_136_) == 0)
{
lean_object* v_a_137_; lean_object* v___x_138_; 
v_a_137_ = lean_ctor_get(v___x_136_, 0);
lean_inc(v_a_137_);
lean_dec_ref_known(v___x_136_, 1);
v___x_138_ = lp_mathlib_Mathlib_Tactic_liftReflToEq(v_a_137_, v___y_129_, v___y_130_, v___y_131_, v___y_132_);
if (lean_obj_tag(v___x_138_) == 0)
{
lean_object* v_a_139_; lean_object* v___x_140_; lean_object* v___x_141_; lean_object* v___x_142_; 
v_a_139_ = lean_ctor_get(v___x_138_, 0);
lean_inc(v_a_139_);
lean_dec_ref_known(v___x_138_, 1);
v___x_140_ = lean_box(0);
v___x_141_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_141_, 0, v_a_139_);
lean_ctor_set(v___x_141_, 1, v___x_140_);
v___x_142_ = l_Lean_Elab_Tactic_replaceMainGoal___redArg(v___x_141_, v___y_126_, v___y_129_, v___y_130_, v___y_131_, v___y_132_);
if (lean_obj_tag(v___x_142_) == 0)
{
lean_object* v___x_144_; uint8_t v_isShared_145_; uint8_t v_isSharedCheck_150_; 
v_isSharedCheck_150_ = !lean_is_exclusive(v___x_142_);
if (v_isSharedCheck_150_ == 0)
{
lean_object* v_unused_151_; 
v_unused_151_ = lean_ctor_get(v___x_142_, 0);
lean_dec(v_unused_151_);
v___x_144_ = v___x_142_;
v_isShared_145_ = v_isSharedCheck_150_;
goto v_resetjp_143_;
}
else
{
lean_dec(v___x_142_);
v___x_144_ = lean_box(0);
v_isShared_145_ = v_isSharedCheck_150_;
goto v_resetjp_143_;
}
v_resetjp_143_:
{
lean_object* v___x_146_; lean_object* v___x_148_; 
v___x_146_ = lean_box(0);
if (v_isShared_145_ == 0)
{
lean_ctor_set(v___x_144_, 0, v___x_146_);
v___x_148_ = v___x_144_;
goto v_reusejp_147_;
}
else
{
lean_object* v_reuseFailAlloc_149_; 
v_reuseFailAlloc_149_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_149_, 0, v___x_146_);
v___x_148_ = v_reuseFailAlloc_149_;
goto v_reusejp_147_;
}
v_reusejp_147_:
{
return v___x_148_;
}
}
}
else
{
return v___x_142_;
}
}
else
{
lean_object* v_a_152_; lean_object* v___x_154_; uint8_t v_isShared_155_; uint8_t v_isSharedCheck_159_; 
v_a_152_ = lean_ctor_get(v___x_138_, 0);
v_isSharedCheck_159_ = !lean_is_exclusive(v___x_138_);
if (v_isSharedCheck_159_ == 0)
{
v___x_154_ = v___x_138_;
v_isShared_155_ = v_isSharedCheck_159_;
goto v_resetjp_153_;
}
else
{
lean_inc(v_a_152_);
lean_dec(v___x_138_);
v___x_154_ = lean_box(0);
v_isShared_155_ = v_isSharedCheck_159_;
goto v_resetjp_153_;
}
v_resetjp_153_:
{
lean_object* v___x_157_; 
if (v_isShared_155_ == 0)
{
v___x_157_ = v___x_154_;
goto v_reusejp_156_;
}
else
{
lean_object* v_reuseFailAlloc_158_; 
v_reuseFailAlloc_158_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_158_, 0, v_a_152_);
v___x_157_ = v_reuseFailAlloc_158_;
goto v_reusejp_156_;
}
v_reusejp_156_:
{
return v___x_157_;
}
}
}
}
else
{
lean_object* v_a_160_; lean_object* v___x_162_; uint8_t v_isShared_163_; uint8_t v_isSharedCheck_167_; 
v_a_160_ = lean_ctor_get(v___x_136_, 0);
v_isSharedCheck_167_ = !lean_is_exclusive(v___x_136_);
if (v_isSharedCheck_167_ == 0)
{
v___x_162_ = v___x_136_;
v_isShared_163_ = v_isSharedCheck_167_;
goto v_resetjp_161_;
}
else
{
lean_inc(v_a_160_);
lean_dec(v___x_136_);
v___x_162_ = lean_box(0);
v_isShared_163_ = v_isSharedCheck_167_;
goto v_resetjp_161_;
}
v_resetjp_161_:
{
lean_object* v___x_165_; 
if (v_isShared_163_ == 0)
{
v___x_165_ = v___x_162_;
goto v_reusejp_164_;
}
else
{
lean_object* v_reuseFailAlloc_166_; 
v_reuseFailAlloc_166_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_166_, 0, v_a_160_);
v___x_165_ = v_reuseFailAlloc_166_;
goto v_reusejp_164_;
}
v_reusejp_164_:
{
return v___x_165_;
}
}
}
}
else
{
lean_object* v_a_168_; lean_object* v___x_170_; uint8_t v_isShared_171_; uint8_t v_isSharedCheck_175_; 
v_a_168_ = lean_ctor_get(v___x_134_, 0);
v_isSharedCheck_175_ = !lean_is_exclusive(v___x_134_);
if (v_isSharedCheck_175_ == 0)
{
v___x_170_ = v___x_134_;
v_isShared_171_ = v_isSharedCheck_175_;
goto v_resetjp_169_;
}
else
{
lean_inc(v_a_168_);
lean_dec(v___x_134_);
v___x_170_ = lean_box(0);
v_isShared_171_ = v_isSharedCheck_175_;
goto v_resetjp_169_;
}
v_resetjp_169_:
{
lean_object* v___x_173_; 
if (v_isShared_171_ == 0)
{
v___x_173_ = v___x_170_;
goto v_reusejp_172_;
}
else
{
lean_object* v_reuseFailAlloc_174_; 
v_reuseFailAlloc_174_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_174_, 0, v_a_168_);
v___x_173_ = v_reuseFailAlloc_174_;
goto v_reusejp_172_;
}
v_reusejp_172_:
{
return v___x_173_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__0___boxed(lean_object* v___y_176_, lean_object* v___y_177_, lean_object* v___y_178_, lean_object* v___y_179_, lean_object* v___y_180_, lean_object* v___y_181_, lean_object* v___y_182_, lean_object* v___y_183_, lean_object* v___y_184_){
_start:
{
lean_object* v_res_185_; 
v_res_185_ = lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__0(v___y_176_, v___y_177_, v___y_178_, v___y_179_, v___y_180_, v___y_181_, v___y_182_, v___y_183_);
lean_dec(v___y_183_);
lean_dec_ref(v___y_182_);
lean_dec(v___y_181_);
lean_dec_ref(v___y_180_);
lean_dec(v___y_179_);
lean_dec_ref(v___y_178_);
lean_dec(v___y_177_);
lean_dec_ref(v___y_176_);
return v_res_185_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___closed__12(void){
_start:
{
lean_object* v___x_207_; lean_object* v___x_208_; 
v___x_207_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___closed__11));
v___x_208_ = l_String_toRawSubstring_x27(v___x_207_);
return v___x_208_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1(lean_object* v___x_247_, lean_object* v___x_248_, lean_object* v_a_249_, lean_object* v___y_250_, lean_object* v___y_251_, lean_object* v___y_252_, lean_object* v___y_253_, lean_object* v___y_254_, lean_object* v___y_255_, lean_object* v___y_256_, lean_object* v___y_257_){
_start:
{
lean_object* v___x_259_; 
v___x_259_ = l_Lean_Elab_Tactic_getMainTarget(v___y_250_, v___y_251_, v___y_252_, v___y_253_, v___y_254_, v___y_255_, v___y_256_, v___y_257_);
if (lean_obj_tag(v___x_259_) == 0)
{
lean_object* v_a_260_; lean_object* v___x_261_; 
v_a_260_ = lean_ctor_get(v___x_259_, 0);
lean_inc(v_a_260_);
lean_dec_ref_known(v___x_259_, 1);
v___x_261_ = l_Lean_Elab_Term_exprToSyntax(v_a_260_, v___y_252_, v___y_253_, v___y_254_, v___y_255_, v___y_256_, v___y_257_);
if (lean_obj_tag(v___x_261_) == 0)
{
lean_object* v_a_262_; lean_object* v_ref_263_; lean_object* v_quotContext_264_; lean_object* v_currMacroScope_265_; uint8_t v___x_266_; lean_object* v___x_267_; lean_object* v___x_268_; lean_object* v___x_269_; lean_object* v___x_270_; lean_object* v___x_271_; lean_object* v___x_272_; lean_object* v___x_273_; lean_object* v___x_274_; lean_object* v___x_275_; lean_object* v___x_276_; lean_object* v___x_277_; lean_object* v___x_278_; lean_object* v___x_279_; lean_object* v___x_280_; lean_object* v___x_281_; lean_object* v___x_282_; lean_object* v___x_283_; lean_object* v___x_284_; lean_object* v___x_285_; lean_object* v___x_286_; lean_object* v___x_287_; lean_object* v___x_288_; lean_object* v___x_289_; lean_object* v___x_290_; lean_object* v___x_291_; lean_object* v___x_292_; lean_object* v___x_293_; lean_object* v___x_294_; lean_object* v___x_295_; lean_object* v___x_296_; lean_object* v___x_297_; lean_object* v___x_298_; lean_object* v___x_299_; lean_object* v___x_300_; lean_object* v___x_301_; lean_object* v___x_302_; lean_object* v___x_303_; lean_object* v___x_304_; lean_object* v___x_305_; lean_object* v___x_306_; lean_object* v___x_307_; lean_object* v___x_308_; lean_object* v___x_309_; lean_object* v___x_310_; lean_object* v___x_311_; lean_object* v___x_312_; 
v_a_262_ = lean_ctor_get(v___x_261_, 0);
lean_inc(v_a_262_);
lean_dec_ref_known(v___x_261_, 1);
v_ref_263_ = lean_ctor_get(v___y_256_, 5);
v_quotContext_264_ = lean_ctor_get(v___y_256_, 10);
v_currMacroScope_265_ = lean_ctor_get(v___y_256_, 11);
v___x_266_ = 0;
v___x_267_ = l_Lean_SourceInfo_fromRef(v_ref_263_, v___x_266_);
v___x_268_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___closed__0));
v___x_269_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___closed__1));
v___x_270_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___closed__2));
lean_inc_ref_n(v___x_247_, 4);
v___x_271_ = l_Lean_Name_mkStr4(v___x_268_, v___x_269_, v___x_247_, v___x_270_);
lean_inc_n(v___x_267_, 11);
v___x_272_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_272_, 0, v___x_267_);
lean_ctor_set(v___x_272_, 1, v___x_270_);
v___x_273_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___closed__5));
v___x_274_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___closed__7));
v___x_275_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___closed__8));
v___x_276_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_276_, 0, v___x_267_);
lean_ctor_set(v___x_276_, 1, v___x_275_);
v___x_277_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___closed__10));
v___x_278_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___closed__12, &lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___closed__12_once, _init_lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___closed__12);
v___x_279_ = lean_box(0);
lean_inc(v_currMacroScope_265_);
lean_inc(v_quotContext_264_);
v___x_280_ = l_Lean_addMacroScope(v_quotContext_264_, v___x_279_, v_currMacroScope_265_);
lean_inc_ref(v___x_248_);
v___x_281_ = l_Lean_Name_mkStr2(v___x_248_, v___x_247_);
v___x_282_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_282_, 0, v___x_281_);
v___x_283_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___closed__15));
v___x_284_ = l_Lean_Name_mkStr3(v___x_268_, v___x_269_, v___x_247_);
v___x_285_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_285_, 0, v___x_284_);
v___x_286_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___closed__16));
v___x_287_ = l_Lean_Name_mkStr3(v___x_268_, v___x_286_, v___x_247_);
v___x_288_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_288_, 0, v___x_287_);
v___x_289_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___closed__25));
lean_inc_ref(v___x_282_);
v___x_290_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_290_, 0, v___x_282_);
lean_ctor_set(v___x_290_, 1, v___x_289_);
v___x_291_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_291_, 0, v___x_288_);
lean_ctor_set(v___x_291_, 1, v___x_290_);
v___x_292_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_292_, 0, v___x_285_);
lean_ctor_set(v___x_292_, 1, v___x_291_);
v___x_293_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_293_, 0, v___x_283_);
lean_ctor_set(v___x_293_, 1, v___x_292_);
v___x_294_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_294_, 0, v___x_282_);
lean_ctor_set(v___x_294_, 1, v___x_293_);
v___x_295_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_295_, 0, v___x_267_);
lean_ctor_set(v___x_295_, 1, v___x_278_);
lean_ctor_set(v___x_295_, 2, v___x_280_);
lean_ctor_set(v___x_295_, 3, v___x_294_);
v___x_296_ = l_Lean_Syntax_node1(v___x_267_, v___x_277_, v___x_295_);
v___x_297_ = l_Lean_Syntax_node2(v___x_267_, v___x_274_, v___x_276_, v___x_296_);
v___x_298_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___closed__26));
v___x_299_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___closed__27));
v___x_300_ = l_Lean_Name_mkStr4(v___x_248_, v___x_247_, v___x_298_, v___x_299_);
v___x_301_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___closed__28));
v___x_302_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_302_, 0, v___x_267_);
lean_ctor_set(v___x_302_, 1, v___x_301_);
v___x_303_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___closed__29));
v___x_304_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_304_, 0, v___x_267_);
lean_ctor_set(v___x_304_, 1, v___x_303_);
lean_inc_ref(v___x_304_);
v___x_305_ = l_Lean_Syntax_node3(v___x_267_, v___x_300_, v___x_302_, v_a_249_, v___x_304_);
v___x_306_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___closed__30));
v___x_307_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_307_, 0, v___x_267_);
lean_ctor_set(v___x_307_, 1, v___x_306_);
v___x_308_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___closed__32));
v___x_309_ = l_Lean_Syntax_node1(v___x_267_, v___x_308_, v_a_262_);
v___x_310_ = l_Lean_Syntax_node5(v___x_267_, v___x_273_, v___x_297_, v___x_305_, v___x_307_, v___x_309_, v___x_304_);
v___x_311_ = l_Lean_Syntax_node2(v___x_267_, v___x_271_, v___x_272_, v___x_310_);
v___x_312_ = l_Lean_Elab_Tactic_evalTactic(v___x_311_, v___y_250_, v___y_251_, v___y_252_, v___y_253_, v___y_254_, v___y_255_, v___y_256_, v___y_257_);
lean_dec_ref(v___y_256_);
return v___x_312_;
}
else
{
lean_object* v_a_313_; lean_object* v___x_315_; uint8_t v_isShared_316_; uint8_t v_isSharedCheck_320_; 
lean_dec_ref(v___y_256_);
lean_dec(v_a_249_);
lean_dec_ref(v___x_248_);
lean_dec_ref(v___x_247_);
v_a_313_ = lean_ctor_get(v___x_261_, 0);
v_isSharedCheck_320_ = !lean_is_exclusive(v___x_261_);
if (v_isSharedCheck_320_ == 0)
{
v___x_315_ = v___x_261_;
v_isShared_316_ = v_isSharedCheck_320_;
goto v_resetjp_314_;
}
else
{
lean_inc(v_a_313_);
lean_dec(v___x_261_);
v___x_315_ = lean_box(0);
v_isShared_316_ = v_isSharedCheck_320_;
goto v_resetjp_314_;
}
v_resetjp_314_:
{
lean_object* v___x_318_; 
if (v_isShared_316_ == 0)
{
v___x_318_ = v___x_315_;
goto v_reusejp_317_;
}
else
{
lean_object* v_reuseFailAlloc_319_; 
v_reuseFailAlloc_319_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_319_, 0, v_a_313_);
v___x_318_ = v_reuseFailAlloc_319_;
goto v_reusejp_317_;
}
v_reusejp_317_:
{
return v___x_318_;
}
}
}
}
else
{
lean_object* v_a_321_; lean_object* v___x_323_; uint8_t v_isShared_324_; uint8_t v_isSharedCheck_328_; 
lean_dec_ref(v___y_256_);
lean_dec(v_a_249_);
lean_dec_ref(v___x_248_);
lean_dec_ref(v___x_247_);
v_a_321_ = lean_ctor_get(v___x_259_, 0);
v_isSharedCheck_328_ = !lean_is_exclusive(v___x_259_);
if (v_isSharedCheck_328_ == 0)
{
v___x_323_ = v___x_259_;
v_isShared_324_ = v_isSharedCheck_328_;
goto v_resetjp_322_;
}
else
{
lean_inc(v_a_321_);
lean_dec(v___x_259_);
v___x_323_ = lean_box(0);
v_isShared_324_ = v_isSharedCheck_328_;
goto v_resetjp_322_;
}
v_resetjp_322_:
{
lean_object* v___x_326_; 
if (v_isShared_324_ == 0)
{
v___x_326_ = v___x_323_;
goto v_reusejp_325_;
}
else
{
lean_object* v_reuseFailAlloc_327_; 
v_reuseFailAlloc_327_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_327_, 0, v_a_321_);
v___x_326_ = v_reuseFailAlloc_327_;
goto v_reusejp_325_;
}
v_reusejp_325_:
{
return v___x_326_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___boxed(lean_object* v___x_329_, lean_object* v___x_330_, lean_object* v_a_331_, lean_object* v___y_332_, lean_object* v___y_333_, lean_object* v___y_334_, lean_object* v___y_335_, lean_object* v___y_336_, lean_object* v___y_337_, lean_object* v___y_338_, lean_object* v___y_339_, lean_object* v___y_340_){
_start:
{
lean_object* v_res_341_; 
v_res_341_ = lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1(v___x_329_, v___x_330_, v_a_331_, v___y_332_, v___y_333_, v___y_334_, v___y_335_, v___y_336_, v___y_337_, v___y_338_, v___y_339_);
lean_dec(v___y_339_);
lean_dec(v___y_337_);
lean_dec_ref(v___y_336_);
lean_dec(v___y_335_);
lean_dec_ref(v___y_334_);
lean_dec(v___y_333_);
lean_dec_ref(v___y_332_);
return v_res_341_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_addTrace___at___00Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1_spec__2_spec__3(lean_object* v_msgData_342_, lean_object* v___y_343_, lean_object* v___y_344_, lean_object* v___y_345_, lean_object* v___y_346_){
_start:
{
lean_object* v___x_348_; lean_object* v_env_349_; lean_object* v___x_350_; lean_object* v_mctx_351_; lean_object* v_lctx_352_; lean_object* v_options_353_; lean_object* v___x_354_; lean_object* v___x_355_; lean_object* v___x_356_; 
v___x_348_ = lean_st_ref_get(v___y_346_);
v_env_349_ = lean_ctor_get(v___x_348_, 0);
lean_inc_ref(v_env_349_);
lean_dec(v___x_348_);
v___x_350_ = lean_st_ref_get(v___y_344_);
v_mctx_351_ = lean_ctor_get(v___x_350_, 0);
lean_inc_ref(v_mctx_351_);
lean_dec(v___x_350_);
v_lctx_352_ = lean_ctor_get(v___y_343_, 2);
v_options_353_ = lean_ctor_get(v___y_345_, 2);
lean_inc_ref(v_options_353_);
lean_inc_ref(v_lctx_352_);
v___x_354_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_354_, 0, v_env_349_);
lean_ctor_set(v___x_354_, 1, v_mctx_351_);
lean_ctor_set(v___x_354_, 2, v_lctx_352_);
lean_ctor_set(v___x_354_, 3, v_options_353_);
v___x_355_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_355_, 0, v___x_354_);
lean_ctor_set(v___x_355_, 1, v_msgData_342_);
v___x_356_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_356_, 0, v___x_355_);
return v___x_356_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_addTrace___at___00Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1_spec__2_spec__3___boxed(lean_object* v_msgData_357_, lean_object* v___y_358_, lean_object* v___y_359_, lean_object* v___y_360_, lean_object* v___y_361_, lean_object* v___y_362_){
_start:
{
lean_object* v_res_363_; 
v_res_363_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_addTrace___at___00Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1_spec__2_spec__3(v_msgData_357_, v___y_358_, v___y_359_, v___y_360_, v___y_361_);
lean_dec(v___y_361_);
lean_dec_ref(v___y_360_);
lean_dec(v___y_359_);
lean_dec_ref(v___y_358_);
return v_res_363_;
}
}
static double _init_lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1_spec__2___redArg___closed__0(void){
_start:
{
lean_object* v___x_364_; double v___x_365_; 
v___x_364_ = lean_unsigned_to_nat(0u);
v___x_365_ = lean_float_of_nat(v___x_364_);
return v___x_365_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1_spec__2___redArg(lean_object* v_cls_368_, lean_object* v_msg_369_, lean_object* v___y_370_, lean_object* v___y_371_, lean_object* v___y_372_, lean_object* v___y_373_){
_start:
{
lean_object* v_ref_375_; lean_object* v___x_376_; lean_object* v_a_377_; lean_object* v___x_379_; uint8_t v_isShared_380_; uint8_t v_isSharedCheck_421_; 
v_ref_375_ = lean_ctor_get(v___y_372_, 5);
v___x_376_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_addTrace___at___00Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1_spec__2_spec__3(v_msg_369_, v___y_370_, v___y_371_, v___y_372_, v___y_373_);
v_a_377_ = lean_ctor_get(v___x_376_, 0);
v_isSharedCheck_421_ = !lean_is_exclusive(v___x_376_);
if (v_isSharedCheck_421_ == 0)
{
v___x_379_ = v___x_376_;
v_isShared_380_ = v_isSharedCheck_421_;
goto v_resetjp_378_;
}
else
{
lean_inc(v_a_377_);
lean_dec(v___x_376_);
v___x_379_ = lean_box(0);
v_isShared_380_ = v_isSharedCheck_421_;
goto v_resetjp_378_;
}
v_resetjp_378_:
{
lean_object* v___x_381_; lean_object* v_traceState_382_; lean_object* v_env_383_; lean_object* v_nextMacroScope_384_; lean_object* v_ngen_385_; lean_object* v_auxDeclNGen_386_; lean_object* v_cache_387_; lean_object* v_messages_388_; lean_object* v_infoState_389_; lean_object* v_snapshotTasks_390_; lean_object* v___x_392_; uint8_t v_isShared_393_; uint8_t v_isSharedCheck_420_; 
v___x_381_ = lean_st_ref_take(v___y_373_);
v_traceState_382_ = lean_ctor_get(v___x_381_, 4);
v_env_383_ = lean_ctor_get(v___x_381_, 0);
v_nextMacroScope_384_ = lean_ctor_get(v___x_381_, 1);
v_ngen_385_ = lean_ctor_get(v___x_381_, 2);
v_auxDeclNGen_386_ = lean_ctor_get(v___x_381_, 3);
v_cache_387_ = lean_ctor_get(v___x_381_, 5);
v_messages_388_ = lean_ctor_get(v___x_381_, 6);
v_infoState_389_ = lean_ctor_get(v___x_381_, 7);
v_snapshotTasks_390_ = lean_ctor_get(v___x_381_, 8);
v_isSharedCheck_420_ = !lean_is_exclusive(v___x_381_);
if (v_isSharedCheck_420_ == 0)
{
v___x_392_ = v___x_381_;
v_isShared_393_ = v_isSharedCheck_420_;
goto v_resetjp_391_;
}
else
{
lean_inc(v_snapshotTasks_390_);
lean_inc(v_infoState_389_);
lean_inc(v_messages_388_);
lean_inc(v_cache_387_);
lean_inc(v_traceState_382_);
lean_inc(v_auxDeclNGen_386_);
lean_inc(v_ngen_385_);
lean_inc(v_nextMacroScope_384_);
lean_inc(v_env_383_);
lean_dec(v___x_381_);
v___x_392_ = lean_box(0);
v_isShared_393_ = v_isSharedCheck_420_;
goto v_resetjp_391_;
}
v_resetjp_391_:
{
uint64_t v_tid_394_; lean_object* v_traces_395_; lean_object* v___x_397_; uint8_t v_isShared_398_; uint8_t v_isSharedCheck_419_; 
v_tid_394_ = lean_ctor_get_uint64(v_traceState_382_, sizeof(void*)*1);
v_traces_395_ = lean_ctor_get(v_traceState_382_, 0);
v_isSharedCheck_419_ = !lean_is_exclusive(v_traceState_382_);
if (v_isSharedCheck_419_ == 0)
{
v___x_397_ = v_traceState_382_;
v_isShared_398_ = v_isSharedCheck_419_;
goto v_resetjp_396_;
}
else
{
lean_inc(v_traces_395_);
lean_dec(v_traceState_382_);
v___x_397_ = lean_box(0);
v_isShared_398_ = v_isSharedCheck_419_;
goto v_resetjp_396_;
}
v_resetjp_396_:
{
lean_object* v___x_399_; double v___x_400_; uint8_t v___x_401_; lean_object* v___x_402_; lean_object* v___x_403_; lean_object* v___x_404_; lean_object* v___x_405_; lean_object* v___x_406_; lean_object* v___x_407_; lean_object* v___x_409_; 
v___x_399_ = lean_box(0);
v___x_400_ = lean_float_once(&lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1_spec__2___redArg___closed__0, &lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1_spec__2___redArg___closed__0_once, _init_lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1_spec__2___redArg___closed__0);
v___x_401_ = 0;
v___x_402_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___closed__11));
v___x_403_ = lean_alloc_ctor(0, 3, 17);
lean_ctor_set(v___x_403_, 0, v_cls_368_);
lean_ctor_set(v___x_403_, 1, v___x_399_);
lean_ctor_set(v___x_403_, 2, v___x_402_);
lean_ctor_set_float(v___x_403_, sizeof(void*)*3, v___x_400_);
lean_ctor_set_float(v___x_403_, sizeof(void*)*3 + 8, v___x_400_);
lean_ctor_set_uint8(v___x_403_, sizeof(void*)*3 + 16, v___x_401_);
v___x_404_ = ((lean_object*)(lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1_spec__2___redArg___closed__1));
v___x_405_ = lean_alloc_ctor(9, 3, 0);
lean_ctor_set(v___x_405_, 0, v___x_403_);
lean_ctor_set(v___x_405_, 1, v_a_377_);
lean_ctor_set(v___x_405_, 2, v___x_404_);
lean_inc(v_ref_375_);
v___x_406_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_406_, 0, v_ref_375_);
lean_ctor_set(v___x_406_, 1, v___x_405_);
v___x_407_ = l_Lean_PersistentArray_push___redArg(v_traces_395_, v___x_406_);
if (v_isShared_398_ == 0)
{
lean_ctor_set(v___x_397_, 0, v___x_407_);
v___x_409_ = v___x_397_;
goto v_reusejp_408_;
}
else
{
lean_object* v_reuseFailAlloc_418_; 
v_reuseFailAlloc_418_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v_reuseFailAlloc_418_, 0, v___x_407_);
lean_ctor_set_uint64(v_reuseFailAlloc_418_, sizeof(void*)*1, v_tid_394_);
v___x_409_ = v_reuseFailAlloc_418_;
goto v_reusejp_408_;
}
v_reusejp_408_:
{
lean_object* v___x_411_; 
if (v_isShared_393_ == 0)
{
lean_ctor_set(v___x_392_, 4, v___x_409_);
v___x_411_ = v___x_392_;
goto v_reusejp_410_;
}
else
{
lean_object* v_reuseFailAlloc_417_; 
v_reuseFailAlloc_417_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_417_, 0, v_env_383_);
lean_ctor_set(v_reuseFailAlloc_417_, 1, v_nextMacroScope_384_);
lean_ctor_set(v_reuseFailAlloc_417_, 2, v_ngen_385_);
lean_ctor_set(v_reuseFailAlloc_417_, 3, v_auxDeclNGen_386_);
lean_ctor_set(v_reuseFailAlloc_417_, 4, v___x_409_);
lean_ctor_set(v_reuseFailAlloc_417_, 5, v_cache_387_);
lean_ctor_set(v_reuseFailAlloc_417_, 6, v_messages_388_);
lean_ctor_set(v_reuseFailAlloc_417_, 7, v_infoState_389_);
lean_ctor_set(v_reuseFailAlloc_417_, 8, v_snapshotTasks_390_);
v___x_411_ = v_reuseFailAlloc_417_;
goto v_reusejp_410_;
}
v_reusejp_410_:
{
lean_object* v___x_412_; lean_object* v___x_413_; lean_object* v___x_415_; 
v___x_412_ = lean_st_ref_set(v___y_373_, v___x_411_);
v___x_413_ = lean_box(0);
if (v_isShared_380_ == 0)
{
lean_ctor_set(v___x_379_, 0, v___x_413_);
v___x_415_ = v___x_379_;
goto v_reusejp_414_;
}
else
{
lean_object* v_reuseFailAlloc_416_; 
v_reuseFailAlloc_416_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_416_, 0, v___x_413_);
v___x_415_ = v_reuseFailAlloc_416_;
goto v_reusejp_414_;
}
v_reusejp_414_:
{
return v___x_415_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1_spec__2___redArg___boxed(lean_object* v_cls_422_, lean_object* v_msg_423_, lean_object* v___y_424_, lean_object* v___y_425_, lean_object* v___y_426_, lean_object* v___y_427_, lean_object* v___y_428_){
_start:
{
lean_object* v_res_429_; 
v_res_429_ = lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1_spec__2___redArg(v_cls_422_, v_msg_423_, v___y_424_, v___y_425_, v___y_426_, v___y_427_);
lean_dec(v___y_427_);
lean_dec_ref(v___y_426_);
lean_dec(v___y_425_);
lean_dec_ref(v___y_424_);
return v_res_429_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Syntax_replaceM___at___00Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1_spec__1___lam__0(lean_object* v___x_436_, lean_object* v___x_437_, lean_object* v_stx_438_, lean_object* v___y_439_, lean_object* v___y_440_, lean_object* v___y_441_, lean_object* v___y_442_, lean_object* v___y_443_, lean_object* v___y_444_, lean_object* v___y_445_, lean_object* v___y_446_){
_start:
{
lean_object* v___x_448_; uint8_t v___x_449_; 
v___x_448_ = ((lean_object*)(lp_mathlib_Lean_Syntax_replaceM___at___00Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1_spec__1___lam__0___closed__1));
lean_inc(v_stx_438_);
v___x_449_ = l_Lean_Syntax_isOfKind(v_stx_438_, v___x_448_);
if (v___x_449_ == 0)
{
uint8_t v___x_450_; 
lean_dec(v___x_437_);
lean_dec(v___x_436_);
lean_inc(v_stx_438_);
v___x_450_ = l_Lean_Syntax_isAntiquots(v_stx_438_);
if (v___x_450_ == 0)
{
lean_object* v___x_451_; lean_object* v___x_452_; 
lean_dec(v_stx_438_);
v___x_451_ = lean_box(0);
v___x_452_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_452_, 0, v___x_451_);
return v___x_452_;
}
else
{
lean_object* v___x_453_; lean_object* v___x_454_; 
v___x_453_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_453_, 0, v_stx_438_);
v___x_454_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_454_, 0, v___x_453_);
return v___x_454_;
}
}
else
{
lean_object* v___x_455_; uint8_t v___x_456_; lean_object* v___x_457_; lean_object* v___x_458_; lean_object* v___x_459_; 
v___x_455_ = lean_box(0);
v___x_456_ = 0;
v___x_457_ = l_Lean_Syntax_mkAntiquotNode(v___x_436_, v_stx_438_, v___x_437_, v___x_455_, v___x_456_);
v___x_458_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_458_, 0, v___x_457_);
v___x_459_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_459_, 0, v___x_458_);
return v___x_459_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Syntax_replaceM___at___00Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1_spec__1___lam__0___boxed(lean_object* v___x_460_, lean_object* v___x_461_, lean_object* v_stx_462_, lean_object* v___y_463_, lean_object* v___y_464_, lean_object* v___y_465_, lean_object* v___y_466_, lean_object* v___y_467_, lean_object* v___y_468_, lean_object* v___y_469_, lean_object* v___y_470_, lean_object* v___y_471_){
_start:
{
lean_object* v_res_472_; 
v_res_472_ = lp_mathlib_Lean_Syntax_replaceM___at___00Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1_spec__1___lam__0(v___x_460_, v___x_461_, v_stx_462_, v___y_463_, v___y_464_, v___y_465_, v___y_466_, v___y_467_, v___y_468_, v___y_469_, v___y_470_);
lean_dec(v___y_470_);
lean_dec_ref(v___y_469_);
lean_dec(v___y_468_);
lean_dec_ref(v___y_467_);
lean_dec(v___y_466_);
lean_dec_ref(v___y_465_);
lean_dec(v___y_464_);
lean_dec_ref(v___y_463_);
return v_res_472_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Syntax_replaceM___at___00Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1_spec__1(lean_object* v_x_473_, lean_object* v___y_474_, lean_object* v___y_475_, lean_object* v___y_476_, lean_object* v___y_477_, lean_object* v___y_478_, lean_object* v___y_479_, lean_object* v___y_480_, lean_object* v___y_481_){
_start:
{
lean_object* v___x_483_; lean_object* v___x_484_; 
v___x_483_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_congrM___closed__7));
v___x_484_ = lean_unsigned_to_nat(0u);
if (lean_obj_tag(v_x_473_) == 1)
{
lean_object* v_info_485_; lean_object* v_kind_486_; lean_object* v_args_487_; lean_object* v___x_488_; 
v_info_485_ = lean_ctor_get(v_x_473_, 0);
lean_inc(v_info_485_);
v_kind_486_ = lean_ctor_get(v_x_473_, 1);
lean_inc(v_kind_486_);
v_args_487_ = lean_ctor_get(v_x_473_, 2);
lean_inc_ref(v_args_487_);
v___x_488_ = lp_mathlib_Lean_Syntax_replaceM___at___00Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1_spec__1___lam__0(v___x_483_, v___x_484_, v_x_473_, v___y_474_, v___y_475_, v___y_476_, v___y_477_, v___y_478_, v___y_479_, v___y_480_, v___y_481_);
if (lean_obj_tag(v___x_488_) == 0)
{
lean_object* v_a_489_; lean_object* v___x_491_; uint8_t v_isShared_492_; uint8_t v_isSharedCheck_517_; 
v_a_489_ = lean_ctor_get(v___x_488_, 0);
v_isSharedCheck_517_ = !lean_is_exclusive(v___x_488_);
if (v_isSharedCheck_517_ == 0)
{
v___x_491_ = v___x_488_;
v_isShared_492_ = v_isSharedCheck_517_;
goto v_resetjp_490_;
}
else
{
lean_inc(v_a_489_);
lean_dec(v___x_488_);
v___x_491_ = lean_box(0);
v_isShared_492_ = v_isSharedCheck_517_;
goto v_resetjp_490_;
}
v_resetjp_490_:
{
if (lean_obj_tag(v_a_489_) == 0)
{
size_t v_sz_493_; size_t v___x_494_; lean_object* v___x_495_; 
lean_del_object(v___x_491_);
v_sz_493_ = lean_array_size(v_args_487_);
v___x_494_ = ((size_t)0ULL);
v___x_495_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Syntax_replaceM___at___00Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1_spec__1_spec__1(v_sz_493_, v___x_494_, v_args_487_, v___y_474_, v___y_475_, v___y_476_, v___y_477_, v___y_478_, v___y_479_, v___y_480_, v___y_481_);
if (lean_obj_tag(v___x_495_) == 0)
{
lean_object* v_a_496_; lean_object* v___x_498_; uint8_t v_isShared_499_; uint8_t v_isSharedCheck_504_; 
v_a_496_ = lean_ctor_get(v___x_495_, 0);
v_isSharedCheck_504_ = !lean_is_exclusive(v___x_495_);
if (v_isSharedCheck_504_ == 0)
{
v___x_498_ = v___x_495_;
v_isShared_499_ = v_isSharedCheck_504_;
goto v_resetjp_497_;
}
else
{
lean_inc(v_a_496_);
lean_dec(v___x_495_);
v___x_498_ = lean_box(0);
v_isShared_499_ = v_isSharedCheck_504_;
goto v_resetjp_497_;
}
v_resetjp_497_:
{
lean_object* v___x_500_; lean_object* v___x_502_; 
v___x_500_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_500_, 0, v_info_485_);
lean_ctor_set(v___x_500_, 1, v_kind_486_);
lean_ctor_set(v___x_500_, 2, v_a_496_);
if (v_isShared_499_ == 0)
{
lean_ctor_set(v___x_498_, 0, v___x_500_);
v___x_502_ = v___x_498_;
goto v_reusejp_501_;
}
else
{
lean_object* v_reuseFailAlloc_503_; 
v_reuseFailAlloc_503_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_503_, 0, v___x_500_);
v___x_502_ = v_reuseFailAlloc_503_;
goto v_reusejp_501_;
}
v_reusejp_501_:
{
return v___x_502_;
}
}
}
else
{
lean_object* v_a_505_; lean_object* v___x_507_; uint8_t v_isShared_508_; uint8_t v_isSharedCheck_512_; 
lean_dec(v_kind_486_);
lean_dec(v_info_485_);
v_a_505_ = lean_ctor_get(v___x_495_, 0);
v_isSharedCheck_512_ = !lean_is_exclusive(v___x_495_);
if (v_isSharedCheck_512_ == 0)
{
v___x_507_ = v___x_495_;
v_isShared_508_ = v_isSharedCheck_512_;
goto v_resetjp_506_;
}
else
{
lean_inc(v_a_505_);
lean_dec(v___x_495_);
v___x_507_ = lean_box(0);
v_isShared_508_ = v_isSharedCheck_512_;
goto v_resetjp_506_;
}
v_resetjp_506_:
{
lean_object* v___x_510_; 
if (v_isShared_508_ == 0)
{
v___x_510_ = v___x_507_;
goto v_reusejp_509_;
}
else
{
lean_object* v_reuseFailAlloc_511_; 
v_reuseFailAlloc_511_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_511_, 0, v_a_505_);
v___x_510_ = v_reuseFailAlloc_511_;
goto v_reusejp_509_;
}
v_reusejp_509_:
{
return v___x_510_;
}
}
}
}
else
{
lean_object* v_val_513_; lean_object* v___x_515_; 
lean_dec_ref(v_args_487_);
lean_dec(v_kind_486_);
lean_dec(v_info_485_);
v_val_513_ = lean_ctor_get(v_a_489_, 0);
lean_inc(v_val_513_);
lean_dec_ref_known(v_a_489_, 1);
if (v_isShared_492_ == 0)
{
lean_ctor_set(v___x_491_, 0, v_val_513_);
v___x_515_ = v___x_491_;
goto v_reusejp_514_;
}
else
{
lean_object* v_reuseFailAlloc_516_; 
v_reuseFailAlloc_516_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_516_, 0, v_val_513_);
v___x_515_ = v_reuseFailAlloc_516_;
goto v_reusejp_514_;
}
v_reusejp_514_:
{
return v___x_515_;
}
}
}
}
else
{
lean_object* v_a_518_; lean_object* v___x_520_; uint8_t v_isShared_521_; uint8_t v_isSharedCheck_525_; 
lean_dec_ref(v_args_487_);
lean_dec(v_kind_486_);
lean_dec(v_info_485_);
v_a_518_ = lean_ctor_get(v___x_488_, 0);
v_isSharedCheck_525_ = !lean_is_exclusive(v___x_488_);
if (v_isSharedCheck_525_ == 0)
{
v___x_520_ = v___x_488_;
v_isShared_521_ = v_isSharedCheck_525_;
goto v_resetjp_519_;
}
else
{
lean_inc(v_a_518_);
lean_dec(v___x_488_);
v___x_520_ = lean_box(0);
v_isShared_521_ = v_isSharedCheck_525_;
goto v_resetjp_519_;
}
v_resetjp_519_:
{
lean_object* v___x_523_; 
if (v_isShared_521_ == 0)
{
v___x_523_ = v___x_520_;
goto v_reusejp_522_;
}
else
{
lean_object* v_reuseFailAlloc_524_; 
v_reuseFailAlloc_524_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_524_, 0, v_a_518_);
v___x_523_ = v_reuseFailAlloc_524_;
goto v_reusejp_522_;
}
v_reusejp_522_:
{
return v___x_523_;
}
}
}
}
else
{
lean_object* v___x_526_; 
lean_inc(v_x_473_);
v___x_526_ = lp_mathlib_Lean_Syntax_replaceM___at___00Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1_spec__1___lam__0(v___x_483_, v___x_484_, v_x_473_, v___y_474_, v___y_475_, v___y_476_, v___y_477_, v___y_478_, v___y_479_, v___y_480_, v___y_481_);
if (lean_obj_tag(v___x_526_) == 0)
{
lean_object* v_a_527_; lean_object* v___x_529_; uint8_t v_isShared_530_; uint8_t v_isSharedCheck_538_; 
v_a_527_ = lean_ctor_get(v___x_526_, 0);
v_isSharedCheck_538_ = !lean_is_exclusive(v___x_526_);
if (v_isSharedCheck_538_ == 0)
{
v___x_529_ = v___x_526_;
v_isShared_530_ = v_isSharedCheck_538_;
goto v_resetjp_528_;
}
else
{
lean_inc(v_a_527_);
lean_dec(v___x_526_);
v___x_529_ = lean_box(0);
v_isShared_530_ = v_isSharedCheck_538_;
goto v_resetjp_528_;
}
v_resetjp_528_:
{
if (lean_obj_tag(v_a_527_) == 0)
{
lean_object* v___x_532_; 
if (v_isShared_530_ == 0)
{
lean_ctor_set(v___x_529_, 0, v_x_473_);
v___x_532_ = v___x_529_;
goto v_reusejp_531_;
}
else
{
lean_object* v_reuseFailAlloc_533_; 
v_reuseFailAlloc_533_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_533_, 0, v_x_473_);
v___x_532_ = v_reuseFailAlloc_533_;
goto v_reusejp_531_;
}
v_reusejp_531_:
{
return v___x_532_;
}
}
else
{
lean_object* v_val_534_; lean_object* v___x_536_; 
lean_dec(v_x_473_);
v_val_534_ = lean_ctor_get(v_a_527_, 0);
lean_inc(v_val_534_);
lean_dec_ref_known(v_a_527_, 1);
if (v_isShared_530_ == 0)
{
lean_ctor_set(v___x_529_, 0, v_val_534_);
v___x_536_ = v___x_529_;
goto v_reusejp_535_;
}
else
{
lean_object* v_reuseFailAlloc_537_; 
v_reuseFailAlloc_537_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_537_, 0, v_val_534_);
v___x_536_ = v_reuseFailAlloc_537_;
goto v_reusejp_535_;
}
v_reusejp_535_:
{
return v___x_536_;
}
}
}
}
else
{
lean_object* v_a_539_; lean_object* v___x_541_; uint8_t v_isShared_542_; uint8_t v_isSharedCheck_546_; 
lean_dec(v_x_473_);
v_a_539_ = lean_ctor_get(v___x_526_, 0);
v_isSharedCheck_546_ = !lean_is_exclusive(v___x_526_);
if (v_isSharedCheck_546_ == 0)
{
v___x_541_ = v___x_526_;
v_isShared_542_ = v_isSharedCheck_546_;
goto v_resetjp_540_;
}
else
{
lean_inc(v_a_539_);
lean_dec(v___x_526_);
v___x_541_ = lean_box(0);
v_isShared_542_ = v_isSharedCheck_546_;
goto v_resetjp_540_;
}
v_resetjp_540_:
{
lean_object* v___x_544_; 
if (v_isShared_542_ == 0)
{
v___x_544_ = v___x_541_;
goto v_reusejp_543_;
}
else
{
lean_object* v_reuseFailAlloc_545_; 
v_reuseFailAlloc_545_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_545_, 0, v_a_539_);
v___x_544_ = v_reuseFailAlloc_545_;
goto v_reusejp_543_;
}
v_reusejp_543_:
{
return v___x_544_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Syntax_replaceM___at___00Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1_spec__1_spec__1(size_t v_sz_547_, size_t v_i_548_, lean_object* v_bs_549_, lean_object* v___y_550_, lean_object* v___y_551_, lean_object* v___y_552_, lean_object* v___y_553_, lean_object* v___y_554_, lean_object* v___y_555_, lean_object* v___y_556_, lean_object* v___y_557_){
_start:
{
uint8_t v___x_559_; 
v___x_559_ = lean_usize_dec_lt(v_i_548_, v_sz_547_);
if (v___x_559_ == 0)
{
lean_object* v___x_560_; 
v___x_560_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_560_, 0, v_bs_549_);
return v___x_560_;
}
else
{
lean_object* v_v_561_; lean_object* v___x_562_; 
v_v_561_ = lean_array_uget_borrowed(v_bs_549_, v_i_548_);
lean_inc(v_v_561_);
v___x_562_ = lp_mathlib_Lean_Syntax_replaceM___at___00Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1_spec__1(v_v_561_, v___y_550_, v___y_551_, v___y_552_, v___y_553_, v___y_554_, v___y_555_, v___y_556_, v___y_557_);
if (lean_obj_tag(v___x_562_) == 0)
{
lean_object* v_a_563_; lean_object* v___x_564_; lean_object* v_bs_x27_565_; size_t v___x_566_; size_t v___x_567_; lean_object* v___x_568_; 
v_a_563_ = lean_ctor_get(v___x_562_, 0);
lean_inc(v_a_563_);
lean_dec_ref_known(v___x_562_, 1);
v___x_564_ = lean_unsigned_to_nat(0u);
v_bs_x27_565_ = lean_array_uset(v_bs_549_, v_i_548_, v___x_564_);
v___x_566_ = ((size_t)1ULL);
v___x_567_ = lean_usize_add(v_i_548_, v___x_566_);
v___x_568_ = lean_array_uset(v_bs_x27_565_, v_i_548_, v_a_563_);
v_i_548_ = v___x_567_;
v_bs_549_ = v___x_568_;
goto _start;
}
else
{
lean_object* v_a_570_; lean_object* v___x_572_; uint8_t v_isShared_573_; uint8_t v_isSharedCheck_577_; 
lean_dec_ref(v_bs_549_);
v_a_570_ = lean_ctor_get(v___x_562_, 0);
v_isSharedCheck_577_ = !lean_is_exclusive(v___x_562_);
if (v_isSharedCheck_577_ == 0)
{
v___x_572_ = v___x_562_;
v_isShared_573_ = v_isSharedCheck_577_;
goto v_resetjp_571_;
}
else
{
lean_inc(v_a_570_);
lean_dec(v___x_562_);
v___x_572_ = lean_box(0);
v_isShared_573_ = v_isSharedCheck_577_;
goto v_resetjp_571_;
}
v_resetjp_571_:
{
lean_object* v___x_575_; 
if (v_isShared_573_ == 0)
{
v___x_575_ = v___x_572_;
goto v_reusejp_574_;
}
else
{
lean_object* v_reuseFailAlloc_576_; 
v_reuseFailAlloc_576_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_576_, 0, v_a_570_);
v___x_575_ = v_reuseFailAlloc_576_;
goto v_reusejp_574_;
}
v_reusejp_574_:
{
return v___x_575_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Syntax_replaceM___at___00Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1_spec__1_spec__1___boxed(lean_object* v_sz_578_, lean_object* v_i_579_, lean_object* v_bs_580_, lean_object* v___y_581_, lean_object* v___y_582_, lean_object* v___y_583_, lean_object* v___y_584_, lean_object* v___y_585_, lean_object* v___y_586_, lean_object* v___y_587_, lean_object* v___y_588_, lean_object* v___y_589_){
_start:
{
size_t v_sz_boxed_590_; size_t v_i_boxed_591_; lean_object* v_res_592_; 
v_sz_boxed_590_ = lean_unbox_usize(v_sz_578_);
lean_dec(v_sz_578_);
v_i_boxed_591_ = lean_unbox_usize(v_i_579_);
lean_dec(v_i_579_);
v_res_592_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Syntax_replaceM___at___00Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1_spec__1_spec__1(v_sz_boxed_590_, v_i_boxed_591_, v_bs_580_, v___y_581_, v___y_582_, v___y_583_, v___y_584_, v___y_585_, v___y_586_, v___y_587_, v___y_588_);
lean_dec(v___y_588_);
lean_dec_ref(v___y_587_);
lean_dec(v___y_586_);
lean_dec_ref(v___y_585_);
lean_dec(v___y_584_);
lean_dec_ref(v___y_583_);
lean_dec(v___y_582_);
lean_dec_ref(v___y_581_);
return v_res_592_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Syntax_replaceM___at___00Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1_spec__1___boxed(lean_object* v_x_593_, lean_object* v___y_594_, lean_object* v___y_595_, lean_object* v___y_596_, lean_object* v___y_597_, lean_object* v___y_598_, lean_object* v___y_599_, lean_object* v___y_600_, lean_object* v___y_601_, lean_object* v___y_602_){
_start:
{
lean_object* v_res_603_; 
v_res_603_ = lp_mathlib_Lean_Syntax_replaceM___at___00Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1_spec__1(v_x_593_, v___y_594_, v___y_595_, v___y_596_, v___y_597_, v___y_598_, v___y_599_, v___y_600_, v___y_601_);
lean_dec(v___y_601_);
lean_dec_ref(v___y_600_);
lean_dec(v___y_599_);
lean_dec_ref(v___y_598_);
lean_dec(v___y_597_);
lean_dec_ref(v___y_596_);
lean_dec(v___y_595_);
lean_dec_ref(v___y_594_);
return v_res_603_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___closed__3(void){
_start:
{
lean_object* v___x_608_; lean_object* v___x_609_; lean_object* v___x_610_; 
v___x_608_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_CongrM_0__Mathlib_Tactic_initFn___closed__2_00___x40_Mathlib_Tactic_CongrM_1572680169____hygCtx___hyg_2_));
v___x_609_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___closed__2));
v___x_610_ = l_Lean_Name_append(v___x_609_, v___x_608_);
return v___x_610_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___closed__5(void){
_start:
{
lean_object* v___x_612_; lean_object* v___x_613_; 
v___x_612_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___closed__4));
v___x_613_ = l_Lean_stringToMessageData(v___x_612_);
return v___x_613_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1(lean_object* v_x_614_, lean_object* v_a_615_, lean_object* v_a_616_, lean_object* v_a_617_, lean_object* v_a_618_, lean_object* v_a_619_, lean_object* v_a_620_, lean_object* v_a_621_, lean_object* v_a_622_){
_start:
{
lean_object* v___x_624_; lean_object* v___x_625_; lean_object* v___x_626_; uint8_t v___x_627_; 
v___x_624_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_CongrM_0__Mathlib_Tactic_initFn___closed__5_00___x40_Mathlib_Tactic_CongrM_1572680169____hygCtx___hyg_2_));
v___x_625_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_CongrM_0__Mathlib_Tactic_initFn___closed__0_00___x40_Mathlib_Tactic_CongrM_1572680169____hygCtx___hyg_2_));
v___x_626_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_congrM___closed__1));
lean_inc(v_x_614_);
v___x_627_ = l_Lean_Syntax_isOfKind(v_x_614_, v___x_626_);
if (v___x_627_ == 0)
{
lean_object* v___x_628_; 
lean_dec(v_x_614_);
v___x_628_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1_spec__0___redArg();
return v___x_628_;
}
else
{
lean_object* v___x_629_; lean_object* v___x_630_; lean_object* v___x_631_; 
v___x_629_ = lean_unsigned_to_nat(1u);
v___x_630_ = l_Lean_Syntax_getArg(v_x_614_, v___x_629_);
lean_dec(v_x_614_);
v___x_631_ = lp_mathlib_Lean_Syntax_replaceM___at___00Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1_spec__1(v___x_630_, v_a_615_, v_a_616_, v_a_617_, v_a_618_, v_a_619_, v_a_620_, v_a_621_, v_a_622_);
if (lean_obj_tag(v___x_631_) == 0)
{
lean_object* v_options_632_; lean_object* v_a_633_; lean_object* v_inheritedTraceOptions_634_; uint8_t v_hasTrace_635_; lean_object* v___f_636_; lean_object* v___f_637_; lean_object* v___y_639_; lean_object* v___y_640_; lean_object* v___y_641_; lean_object* v___y_642_; lean_object* v___y_643_; lean_object* v___y_644_; lean_object* v___y_645_; lean_object* v___y_646_; 
v_options_632_ = lean_ctor_get(v_a_621_, 2);
v_a_633_ = lean_ctor_get(v___x_631_, 0);
lean_inc_n(v_a_633_, 2);
lean_dec_ref_known(v___x_631_, 1);
v_inheritedTraceOptions_634_ = lean_ctor_get(v_a_621_, 13);
v_hasTrace_635_ = lean_ctor_get_uint8(v_options_632_, sizeof(void*)*1);
v___f_636_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___closed__0));
v___f_637_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___lam__1___boxed), 12, 3);
lean_closure_set(v___f_637_, 0, v___x_625_);
lean_closure_set(v___f_637_, 1, v___x_624_);
lean_closure_set(v___f_637_, 2, v_a_633_);
if (v_hasTrace_635_ == 0)
{
lean_dec(v_a_633_);
v___y_639_ = v_a_615_;
v___y_640_ = v_a_616_;
v___y_641_ = v_a_617_;
v___y_642_ = v_a_618_;
v___y_643_ = v_a_619_;
v___y_644_ = v_a_620_;
v___y_645_ = v_a_621_;
v___y_646_ = v_a_622_;
goto v___jp_638_;
}
else
{
lean_object* v___x_649_; lean_object* v___x_650_; uint8_t v___x_651_; 
v___x_649_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_CongrM_0__Mathlib_Tactic_initFn___closed__2_00___x40_Mathlib_Tactic_CongrM_1572680169____hygCtx___hyg_2_));
v___x_650_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___closed__3, &lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___closed__3_once, _init_lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___closed__3);
v___x_651_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_634_, v_options_632_, v___x_650_);
if (v___x_651_ == 0)
{
lean_dec(v_a_633_);
v___y_639_ = v_a_615_;
v___y_640_ = v_a_616_;
v___y_641_ = v_a_617_;
v___y_642_ = v_a_618_;
v___y_643_ = v_a_619_;
v___y_644_ = v_a_620_;
v___y_645_ = v_a_621_;
v___y_646_ = v_a_622_;
goto v___jp_638_;
}
else
{
lean_object* v___x_652_; lean_object* v___x_653_; lean_object* v___x_654_; lean_object* v___x_655_; 
v___x_652_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___closed__5, &lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___closed__5_once, _init_lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___closed__5);
v___x_653_ = l_Lean_MessageData_ofSyntax(v_a_633_);
v___x_654_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_654_, 0, v___x_652_);
lean_ctor_set(v___x_654_, 1, v___x_653_);
v___x_655_ = lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1_spec__2___redArg(v___x_649_, v___x_654_, v_a_619_, v_a_620_, v_a_621_, v_a_622_);
if (lean_obj_tag(v___x_655_) == 0)
{
lean_dec_ref_known(v___x_655_, 1);
v___y_639_ = v_a_615_;
v___y_640_ = v_a_616_;
v___y_641_ = v_a_617_;
v___y_642_ = v_a_618_;
v___y_643_ = v_a_619_;
v___y_644_ = v_a_620_;
v___y_645_ = v_a_621_;
v___y_646_ = v_a_622_;
goto v___jp_638_;
}
else
{
lean_dec_ref(v___f_637_);
return v___x_655_;
}
}
}
v___jp_638_:
{
lean_object* v___x_647_; 
v___x_647_ = l_Lean_Elab_Tactic_withMainContext___redArg(v___f_636_, v___y_639_, v___y_640_, v___y_641_, v___y_642_, v___y_643_, v___y_644_, v___y_645_, v___y_646_);
if (lean_obj_tag(v___x_647_) == 0)
{
lean_object* v___x_648_; 
lean_dec_ref_known(v___x_647_, 1);
v___x_648_ = l_Lean_Elab_Tactic_withMainContext___redArg(v___f_637_, v___y_639_, v___y_640_, v___y_641_, v___y_642_, v___y_643_, v___y_644_, v___y_645_, v___y_646_);
return v___x_648_;
}
else
{
lean_dec_ref(v___f_637_);
return v___x_647_;
}
}
}
else
{
lean_object* v_a_656_; lean_object* v___x_658_; uint8_t v_isShared_659_; uint8_t v_isSharedCheck_663_; 
v_a_656_ = lean_ctor_get(v___x_631_, 0);
v_isSharedCheck_663_ = !lean_is_exclusive(v___x_631_);
if (v_isSharedCheck_663_ == 0)
{
v___x_658_ = v___x_631_;
v_isShared_659_ = v_isSharedCheck_663_;
goto v_resetjp_657_;
}
else
{
lean_inc(v_a_656_);
lean_dec(v___x_631_);
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
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1___boxed(lean_object* v_x_664_, lean_object* v_a_665_, lean_object* v_a_666_, lean_object* v_a_667_, lean_object* v_a_668_, lean_object* v_a_669_, lean_object* v_a_670_, lean_object* v_a_671_, lean_object* v_a_672_, lean_object* v_a_673_){
_start:
{
lean_object* v_res_674_; 
v_res_674_ = lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1(v_x_664_, v_a_665_, v_a_666_, v_a_667_, v_a_668_, v_a_669_, v_a_670_, v_a_671_, v_a_672_);
lean_dec(v_a_672_);
lean_dec_ref(v_a_671_);
lean_dec(v_a_670_);
lean_dec_ref(v_a_669_);
lean_dec(v_a_668_);
lean_dec_ref(v_a_667_);
lean_dec(v_a_666_);
lean_dec_ref(v_a_665_);
return v_res_674_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1_spec__2(lean_object* v_cls_675_, lean_object* v_msg_676_, lean_object* v___y_677_, lean_object* v___y_678_, lean_object* v___y_679_, lean_object* v___y_680_, lean_object* v___y_681_, lean_object* v___y_682_, lean_object* v___y_683_, lean_object* v___y_684_){
_start:
{
lean_object* v___x_686_; 
v___x_686_ = lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1_spec__2___redArg(v_cls_675_, v_msg_676_, v___y_681_, v___y_682_, v___y_683_, v___y_684_);
return v___x_686_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1_spec__2___boxed(lean_object* v_cls_687_, lean_object* v_msg_688_, lean_object* v___y_689_, lean_object* v___y_690_, lean_object* v___y_691_, lean_object* v___y_692_, lean_object* v___y_693_, lean_object* v___y_694_, lean_object* v___y_695_, lean_object* v___y_696_, lean_object* v___y_697_){
_start:
{
lean_object* v_res_698_; 
v_res_698_ = lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic___aux__Mathlib__Tactic__CongrM______elabRules__Mathlib__Tactic__congrM__1_spec__2(v_cls_687_, v_msg_688_, v___y_689_, v___y_690_, v___y_691_, v___y_692_, v___y_693_, v___y_694_, v___y_695_, v___y_696_);
lean_dec(v___y_696_);
lean_dec_ref(v___y_695_);
lean_dec(v___y_694_);
lean_dec_ref(v___y_693_);
lean_dec(v___y_692_);
lean_dec_ref(v___y_691_);
lean_dec(v___y_690_);
lean_dec_ref(v___y_689_);
return v_res_698_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Relation_Rfl(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_TermCongr(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Tactic_CongrM(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Relation_Rfl(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_TermCongr(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Tactic_CongrM(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = lp_mathlib___private_Mathlib_Tactic_CongrM_0__Mathlib_Tactic_initFn_00___x40_Mathlib_Tactic_CongrM_1572680169____hygCtx___hyg_2_();
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Relation_Rfl(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_TermCongr(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_CongrM(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Relation_Rfl(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_TermCongr(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_CongrM(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Tactic_CongrM(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Tactic_CongrM(builtin);
}
#ifdef __cplusplus
}
#endif
