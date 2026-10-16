// Lean compiler output
// Module: Mathlib.Tactic.AdaptationNote
// Imports: public import Init public meta import Init public meta import Lean.Meta.Tactic.TryThis public meta import Mathlib.Tactic.Linter.Header public import Lean.Meta.TryThis
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
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getOptional_x3f(lean_object*);
lean_object* l_Lean_Name_append(lean_object*, lean_object*);
uint8_t l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_TSyntax_getDocString(lean_object*);
lean_object* l_Lean_MessageData_ofFormat(lean_object*);
lean_object* lean_st_ref_get(lean_object*);
lean_object* lean_st_ref_take(lean_object*);
double lean_float_of_nat(lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* l_Lean_PersistentArray_push___redArg(lean_object*, lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
lean_object* l_Lean_MessageLog_add(lean_object*, lean_object*);
lean_object* l___private_Lean_Log_0__Lean_MessageData_appendDescriptionWidgetIfNamed(lean_object*);
lean_object* l_Lean_FileMap_toPosition(lean_object*, lean_object*);
uint8_t l_Lean_MessageData_hasTag(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getTailPos_x3f(lean_object*, uint8_t);
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getPos_x3f(lean_object*, uint8_t);
uint8_t lean_string_dec_eq(lean_object*, lean_object*);
uint8_t l_Lean_instBEqMessageSeverity_beq(uint8_t, uint8_t);
extern lean_object* l_Lean_warningAsError;
lean_object* l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(lean_object*, lean_object*);
uint8_t l_Lean_MessageData_hasSyntheticSorry(lean_object*);
lean_object* l_Lean_mkAtom(lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_updateTrailing(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_unsetTrailing(lean_object*);
lean_object* l_Lean_Syntax_setArg(lean_object*, lean_object*, lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
extern lean_object* l_Lean_MessageData_nil;
lean_object* l_Lean_Meta_Tactic_TryThis_addSuggestion(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getTailInfo(lean_object*);
extern lean_object* l_Lean_Elab_unsupportedSyntaxExceptionId;
lean_object* l_Lean_Name_str___override(lean_object*, lean_object*);
lean_object* l_Lean_Name_num___override(lean_object*, lean_object*);
lean_object* l_Lean_registerTraceClass(lean_object*, uint8_t, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Elab_Command_liftTermElabM___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Term_elabTerm(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isNone(lean_object*);
uint8_t l_Lean_Syntax_matchesNull(lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_AdaptationNote_0__initFn___closed__0_00___x40_Mathlib_Tactic_AdaptationNote_1330411147____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "adaptationNote"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_AdaptationNote_0__initFn___closed__0_00___x40_Mathlib_Tactic_AdaptationNote_1330411147____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_AdaptationNote_0__initFn___closed__0_00___x40_Mathlib_Tactic_AdaptationNote_1330411147____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_AdaptationNote_0__initFn___closed__1_00___x40_Mathlib_Tactic_AdaptationNote_1330411147____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_AdaptationNote_0__initFn___closed__0_00___x40_Mathlib_Tactic_AdaptationNote_1330411147____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(21, 122, 3, 135, 143, 144, 29, 195)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_AdaptationNote_0__initFn___closed__1_00___x40_Mathlib_Tactic_AdaptationNote_1330411147____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_AdaptationNote_0__initFn___closed__1_00___x40_Mathlib_Tactic_AdaptationNote_1330411147____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_AdaptationNote_0__initFn___closed__2_00___x40_Mathlib_Tactic_AdaptationNote_1330411147____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "_private"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_AdaptationNote_0__initFn___closed__2_00___x40_Mathlib_Tactic_AdaptationNote_1330411147____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_AdaptationNote_0__initFn___closed__2_00___x40_Mathlib_Tactic_AdaptationNote_1330411147____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_AdaptationNote_0__initFn___closed__3_00___x40_Mathlib_Tactic_AdaptationNote_1330411147____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_AdaptationNote_0__initFn___closed__2_00___x40_Mathlib_Tactic_AdaptationNote_1330411147____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(103, 214, 75, 80, 34, 198, 193, 153)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_AdaptationNote_0__initFn___closed__3_00___x40_Mathlib_Tactic_AdaptationNote_1330411147____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_AdaptationNote_0__initFn___closed__3_00___x40_Mathlib_Tactic_AdaptationNote_1330411147____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_AdaptationNote_0__initFn___closed__4_00___x40_Mathlib_Tactic_AdaptationNote_1330411147____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_AdaptationNote_0__initFn___closed__4_00___x40_Mathlib_Tactic_AdaptationNote_1330411147____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_AdaptationNote_0__initFn___closed__4_00___x40_Mathlib_Tactic_AdaptationNote_1330411147____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_AdaptationNote_0__initFn___closed__5_00___x40_Mathlib_Tactic_AdaptationNote_1330411147____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_AdaptationNote_0__initFn___closed__3_00___x40_Mathlib_Tactic_AdaptationNote_1330411147____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_AdaptationNote_0__initFn___closed__4_00___x40_Mathlib_Tactic_AdaptationNote_1330411147____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(234, 232, 174, 134, 127, 136, 69, 92)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_AdaptationNote_0__initFn___closed__5_00___x40_Mathlib_Tactic_AdaptationNote_1330411147____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_AdaptationNote_0__initFn___closed__5_00___x40_Mathlib_Tactic_AdaptationNote_1330411147____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_AdaptationNote_0__initFn___closed__6_00___x40_Mathlib_Tactic_AdaptationNote_1330411147____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_AdaptationNote_0__initFn___closed__6_00___x40_Mathlib_Tactic_AdaptationNote_1330411147____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_AdaptationNote_0__initFn___closed__6_00___x40_Mathlib_Tactic_AdaptationNote_1330411147____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_AdaptationNote_0__initFn___closed__7_00___x40_Mathlib_Tactic_AdaptationNote_1330411147____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_AdaptationNote_0__initFn___closed__5_00___x40_Mathlib_Tactic_AdaptationNote_1330411147____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_AdaptationNote_0__initFn___closed__6_00___x40_Mathlib_Tactic_AdaptationNote_1330411147____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(191, 70, 156, 159, 11, 54, 216, 94)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_AdaptationNote_0__initFn___closed__7_00___x40_Mathlib_Tactic_AdaptationNote_1330411147____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_AdaptationNote_0__initFn___closed__7_00___x40_Mathlib_Tactic_AdaptationNote_1330411147____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_AdaptationNote_0__initFn___closed__8_00___x40_Mathlib_Tactic_AdaptationNote_1330411147____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "AdaptationNote"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_AdaptationNote_0__initFn___closed__8_00___x40_Mathlib_Tactic_AdaptationNote_1330411147____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_AdaptationNote_0__initFn___closed__8_00___x40_Mathlib_Tactic_AdaptationNote_1330411147____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_AdaptationNote_0__initFn___closed__9_00___x40_Mathlib_Tactic_AdaptationNote_1330411147____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_AdaptationNote_0__initFn___closed__7_00___x40_Mathlib_Tactic_AdaptationNote_1330411147____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_AdaptationNote_0__initFn___closed__8_00___x40_Mathlib_Tactic_AdaptationNote_1330411147____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(221, 235, 170, 31, 150, 107, 118, 46)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_AdaptationNote_0__initFn___closed__9_00___x40_Mathlib_Tactic_AdaptationNote_1330411147____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_AdaptationNote_0__initFn___closed__9_00___x40_Mathlib_Tactic_AdaptationNote_1330411147____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_AdaptationNote_0__initFn___closed__10_00___x40_Mathlib_Tactic_AdaptationNote_1330411147____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_AdaptationNote_0__initFn___closed__9_00___x40_Mathlib_Tactic_AdaptationNote_1330411147____hygCtx___hyg_2__value),((lean_object*)(((size_t)(0) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(232, 219, 115, 69, 249, 115, 69, 133)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_AdaptationNote_0__initFn___closed__10_00___x40_Mathlib_Tactic_AdaptationNote_1330411147____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_AdaptationNote_0__initFn___closed__10_00___x40_Mathlib_Tactic_AdaptationNote_1330411147____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_AdaptationNote_0__initFn___closed__11_00___x40_Mathlib_Tactic_AdaptationNote_1330411147____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "initFn"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_AdaptationNote_0__initFn___closed__11_00___x40_Mathlib_Tactic_AdaptationNote_1330411147____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_AdaptationNote_0__initFn___closed__11_00___x40_Mathlib_Tactic_AdaptationNote_1330411147____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_AdaptationNote_0__initFn___closed__12_00___x40_Mathlib_Tactic_AdaptationNote_1330411147____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_AdaptationNote_0__initFn___closed__10_00___x40_Mathlib_Tactic_AdaptationNote_1330411147____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_AdaptationNote_0__initFn___closed__11_00___x40_Mathlib_Tactic_AdaptationNote_1330411147____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(5, 237, 29, 168, 85, 229, 224, 4)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_AdaptationNote_0__initFn___closed__12_00___x40_Mathlib_Tactic_AdaptationNote_1330411147____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_AdaptationNote_0__initFn___closed__12_00___x40_Mathlib_Tactic_AdaptationNote_1330411147____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_AdaptationNote_0__initFn___closed__13_00___x40_Mathlib_Tactic_AdaptationNote_1330411147____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "_@"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_AdaptationNote_0__initFn___closed__13_00___x40_Mathlib_Tactic_AdaptationNote_1330411147____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_AdaptationNote_0__initFn___closed__13_00___x40_Mathlib_Tactic_AdaptationNote_1330411147____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_AdaptationNote_0__initFn___closed__14_00___x40_Mathlib_Tactic_AdaptationNote_1330411147____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_AdaptationNote_0__initFn___closed__12_00___x40_Mathlib_Tactic_AdaptationNote_1330411147____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_AdaptationNote_0__initFn___closed__13_00___x40_Mathlib_Tactic_AdaptationNote_1330411147____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(248, 89, 222, 67, 252, 30, 116, 219)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_AdaptationNote_0__initFn___closed__14_00___x40_Mathlib_Tactic_AdaptationNote_1330411147____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_AdaptationNote_0__initFn___closed__14_00___x40_Mathlib_Tactic_AdaptationNote_1330411147____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_AdaptationNote_0__initFn___closed__15_00___x40_Mathlib_Tactic_AdaptationNote_1330411147____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_AdaptationNote_0__initFn___closed__14_00___x40_Mathlib_Tactic_AdaptationNote_1330411147____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_AdaptationNote_0__initFn___closed__4_00___x40_Mathlib_Tactic_AdaptationNote_1330411147____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(33, 176, 71, 178, 105, 108, 21, 115)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_AdaptationNote_0__initFn___closed__15_00___x40_Mathlib_Tactic_AdaptationNote_1330411147____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_AdaptationNote_0__initFn___closed__15_00___x40_Mathlib_Tactic_AdaptationNote_1330411147____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_AdaptationNote_0__initFn___closed__16_00___x40_Mathlib_Tactic_AdaptationNote_1330411147____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_AdaptationNote_0__initFn___closed__15_00___x40_Mathlib_Tactic_AdaptationNote_1330411147____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_AdaptationNote_0__initFn___closed__6_00___x40_Mathlib_Tactic_AdaptationNote_1330411147____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(104, 190, 8, 39, 247, 85, 65, 5)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_AdaptationNote_0__initFn___closed__16_00___x40_Mathlib_Tactic_AdaptationNote_1330411147____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_AdaptationNote_0__initFn___closed__16_00___x40_Mathlib_Tactic_AdaptationNote_1330411147____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_AdaptationNote_0__initFn___closed__17_00___x40_Mathlib_Tactic_AdaptationNote_1330411147____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_AdaptationNote_0__initFn___closed__16_00___x40_Mathlib_Tactic_AdaptationNote_1330411147____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_AdaptationNote_0__initFn___closed__8_00___x40_Mathlib_Tactic_AdaptationNote_1330411147____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(214, 188, 233, 115, 215, 51, 14, 143)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_AdaptationNote_0__initFn___closed__17_00___x40_Mathlib_Tactic_AdaptationNote_1330411147____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_AdaptationNote_0__initFn___closed__17_00___x40_Mathlib_Tactic_AdaptationNote_1330411147____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_AdaptationNote_0__initFn___closed__18_00___x40_Mathlib_Tactic_AdaptationNote_1330411147____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_AdaptationNote_0__initFn___closed__17_00___x40_Mathlib_Tactic_AdaptationNote_1330411147____hygCtx___hyg_2__value),((lean_object*)(((size_t)(1330411147) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(35, 223, 45, 234, 171, 58, 81, 97)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_AdaptationNote_0__initFn___closed__18_00___x40_Mathlib_Tactic_AdaptationNote_1330411147____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_AdaptationNote_0__initFn___closed__18_00___x40_Mathlib_Tactic_AdaptationNote_1330411147____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_AdaptationNote_0__initFn___closed__19_00___x40_Mathlib_Tactic_AdaptationNote_1330411147____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "_hygCtx"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_AdaptationNote_0__initFn___closed__19_00___x40_Mathlib_Tactic_AdaptationNote_1330411147____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_AdaptationNote_0__initFn___closed__19_00___x40_Mathlib_Tactic_AdaptationNote_1330411147____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_AdaptationNote_0__initFn___closed__20_00___x40_Mathlib_Tactic_AdaptationNote_1330411147____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_AdaptationNote_0__initFn___closed__18_00___x40_Mathlib_Tactic_AdaptationNote_1330411147____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_AdaptationNote_0__initFn___closed__19_00___x40_Mathlib_Tactic_AdaptationNote_1330411147____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(120, 181, 119, 20, 19, 68, 219, 67)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_AdaptationNote_0__initFn___closed__20_00___x40_Mathlib_Tactic_AdaptationNote_1330411147____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_AdaptationNote_0__initFn___closed__20_00___x40_Mathlib_Tactic_AdaptationNote_1330411147____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_AdaptationNote_0__initFn___closed__21_00___x40_Mathlib_Tactic_AdaptationNote_1330411147____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "_hyg"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_AdaptationNote_0__initFn___closed__21_00___x40_Mathlib_Tactic_AdaptationNote_1330411147____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_AdaptationNote_0__initFn___closed__21_00___x40_Mathlib_Tactic_AdaptationNote_1330411147____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_AdaptationNote_0__initFn___closed__22_00___x40_Mathlib_Tactic_AdaptationNote_1330411147____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_AdaptationNote_0__initFn___closed__20_00___x40_Mathlib_Tactic_AdaptationNote_1330411147____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_AdaptationNote_0__initFn___closed__21_00___x40_Mathlib_Tactic_AdaptationNote_1330411147____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(156, 105, 20, 89, 108, 250, 235, 92)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_AdaptationNote_0__initFn___closed__22_00___x40_Mathlib_Tactic_AdaptationNote_1330411147____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_AdaptationNote_0__initFn___closed__22_00___x40_Mathlib_Tactic_AdaptationNote_1330411147____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_AdaptationNote_0__initFn___closed__23_00___x40_Mathlib_Tactic_AdaptationNote_1330411147____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_AdaptationNote_0__initFn___closed__22_00___x40_Mathlib_Tactic_AdaptationNote_1330411147____hygCtx___hyg_2__value),((lean_object*)(((size_t)(2) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(5, 253, 176, 171, 249, 182, 175, 91)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_AdaptationNote_0__initFn___closed__23_00___x40_Mathlib_Tactic_AdaptationNote_1330411147____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_AdaptationNote_0__initFn___closed__23_00___x40_Mathlib_Tactic_AdaptationNote_1330411147____hygCtx___hyg_2__value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_AdaptationNote_0__initFn_00___x40_Mathlib_Tactic_AdaptationNote_1330411147____hygCtx___hyg_2_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_AdaptationNote_0__initFn_00___x40_Mathlib_Tactic_AdaptationNote_1330411147____hygCtx___hyg_2____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_addTrace___at___00reportAdaptationNote_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_addTrace___at___00reportAdaptationNote_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_addTrace___at___00reportAdaptationNote_spec__0___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static double lp_mathlib_Lean_addTrace___at___00reportAdaptationNote_spec__0___closed__0;
static const lean_string_object lp_mathlib_Lean_addTrace___at___00reportAdaptationNote_spec__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1, .m_capacity = 1, .m_length = 0, .m_data = ""};
static const lean_object* lp_mathlib_Lean_addTrace___at___00reportAdaptationNote_spec__0___closed__1 = (const lean_object*)&lp_mathlib_Lean_addTrace___at___00reportAdaptationNote_spec__0___closed__1_value;
static const lean_array_object lp_mathlib_Lean_addTrace___at___00reportAdaptationNote_spec__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Lean_addTrace___at___00reportAdaptationNote_spec__0___closed__2 = (const lean_object*)&lp_mathlib_Lean_addTrace___at___00reportAdaptationNote_spec__0___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00reportAdaptationNote_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00reportAdaptationNote_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logError___at___00reportAdaptationNote_spec__1_spec__2_spec__3___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Elab"};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logError___at___00reportAdaptationNote_spec__1_spec__2_spec__3___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logError___at___00reportAdaptationNote_spec__1_spec__2_spec__3___lam__0___closed__0_value;
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logError___at___00reportAdaptationNote_spec__1_spec__2_spec__3___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "unsolvedGoals"};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logError___at___00reportAdaptationNote_spec__1_spec__2_spec__3___lam__0___closed__1 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logError___at___00reportAdaptationNote_spec__1_spec__2_spec__3___lam__0___closed__1_value;
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logError___at___00reportAdaptationNote_spec__1_spec__2_spec__3___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "synthPlaceholder"};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logError___at___00reportAdaptationNote_spec__1_spec__2_spec__3___lam__0___closed__2 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logError___at___00reportAdaptationNote_spec__1_spec__2_spec__3___lam__0___closed__2_value;
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logError___at___00reportAdaptationNote_spec__1_spec__2_spec__3___lam__0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "lean"};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logError___at___00reportAdaptationNote_spec__1_spec__2_spec__3___lam__0___closed__3 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logError___at___00reportAdaptationNote_spec__1_spec__2_spec__3___lam__0___closed__3_value;
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logError___at___00reportAdaptationNote_spec__1_spec__2_spec__3___lam__0___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "inductionWithNoAlts"};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logError___at___00reportAdaptationNote_spec__1_spec__2_spec__3___lam__0___closed__4 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logError___at___00reportAdaptationNote_spec__1_spec__2_spec__3___lam__0___closed__4_value;
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logError___at___00reportAdaptationNote_spec__1_spec__2_spec__3___lam__0___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "_namedError"};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logError___at___00reportAdaptationNote_spec__1_spec__2_spec__3___lam__0___closed__5 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logError___at___00reportAdaptationNote_spec__1_spec__2_spec__3___lam__0___closed__5_value;
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logError___at___00reportAdaptationNote_spec__1_spec__2_spec__3___lam__0___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "trace"};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logError___at___00reportAdaptationNote_spec__1_spec__2_spec__3___lam__0___closed__6 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logError___at___00reportAdaptationNote_spec__1_spec__2_spec__3___lam__0___closed__6_value;
LEAN_EXPORT uint8_t lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logError___at___00reportAdaptationNote_spec__1_spec__2_spec__3___lam__0(uint8_t, uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logError___at___00reportAdaptationNote_spec__1_spec__2_spec__3___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_log___at___00Lean_logError___at___00reportAdaptationNote_spec__1_spec__2_spec__3_spec__4(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_log___at___00Lean_logError___at___00reportAdaptationNote_spec__1_spec__2_spec__3_spec__4___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logError___at___00reportAdaptationNote_spec__1_spec__2_spec__3(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logError___at___00reportAdaptationNote_spec__1_spec__2_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_log___at___00Lean_logError___at___00reportAdaptationNote_spec__1_spec__2(lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_log___at___00Lean_logError___at___00reportAdaptationNote_spec__1_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logError___at___00reportAdaptationNote_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logError___at___00reportAdaptationNote_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib_reportAdaptationNote___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logError___at___00reportAdaptationNote_spec__1_spec__2_spec__3___lam__0___closed__6_value),LEAN_SCALAR_PTR_LITERAL(212, 145, 141, 177, 67, 149, 127, 197)}};
static const lean_object* lp_mathlib_reportAdaptationNote___closed__0 = (const lean_object*)&lp_mathlib_reportAdaptationNote___closed__0_value;
static lean_once_cell_t lp_mathlib_reportAdaptationNote___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_reportAdaptationNote___closed__1;
static const lean_string_object lp_mathlib_reportAdaptationNote___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 54, .m_capacity = 54, .m_length = 53, .m_data = "Adaptation notes must be followed by a /-- comment -/"};
static const lean_object* lp_mathlib_reportAdaptationNote___closed__2 = (const lean_object*)&lp_mathlib_reportAdaptationNote___closed__2_value;
static const lean_ctor_object lp_mathlib_reportAdaptationNote___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_reportAdaptationNote___closed__2_value)}};
static const lean_object* lp_mathlib_reportAdaptationNote___closed__3 = (const lean_object*)&lp_mathlib_reportAdaptationNote___closed__3_value;
static lean_once_cell_t lp_mathlib_reportAdaptationNote___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_reportAdaptationNote___closed__4;
static const lean_string_object lp_mathlib_reportAdaptationNote___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib_reportAdaptationNote___closed__5 = (const lean_object*)&lp_mathlib_reportAdaptationNote___closed__5_value;
static const lean_string_object lp_mathlib_reportAdaptationNote___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib_reportAdaptationNote___closed__6 = (const lean_object*)&lp_mathlib_reportAdaptationNote___closed__6_value;
static const lean_string_object lp_mathlib_reportAdaptationNote___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Command"};
static const lean_object* lp_mathlib_reportAdaptationNote___closed__7 = (const lean_object*)&lp_mathlib_reportAdaptationNote___closed__7_value;
static const lean_string_object lp_mathlib_reportAdaptationNote___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "docComment"};
static const lean_object* lp_mathlib_reportAdaptationNote___closed__8 = (const lean_object*)&lp_mathlib_reportAdaptationNote___closed__8_value;
static const lean_ctor_object lp_mathlib_reportAdaptationNote___closed__9_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_reportAdaptationNote___closed__5_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_reportAdaptationNote___closed__9_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_reportAdaptationNote___closed__9_value_aux_0),((lean_object*)&lp_mathlib_reportAdaptationNote___closed__6_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_reportAdaptationNote___closed__9_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_reportAdaptationNote___closed__9_value_aux_1),((lean_object*)&lp_mathlib_reportAdaptationNote___closed__7_value),LEAN_SCALAR_PTR_LITERAL(214, 208, 105, 11, 221, 56, 173, 240)}};
static const lean_ctor_object lp_mathlib_reportAdaptationNote___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_reportAdaptationNote___closed__9_value_aux_2),((lean_object*)&lp_mathlib_reportAdaptationNote___closed__8_value),LEAN_SCALAR_PTR_LITERAL(44, 76, 179, 33, 27, 4, 201, 125)}};
static const lean_object* lp_mathlib_reportAdaptationNote___closed__9 = (const lean_object*)&lp_mathlib_reportAdaptationNote___closed__9_value;
static const lean_string_object lp_mathlib_reportAdaptationNote___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "/--"};
static const lean_object* lp_mathlib_reportAdaptationNote___closed__10 = (const lean_object*)&lp_mathlib_reportAdaptationNote___closed__10_value;
static lean_once_cell_t lp_mathlib_reportAdaptationNote___closed__11_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_reportAdaptationNote___closed__11;
static const lean_string_object lp_mathlib_reportAdaptationNote___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "comment -/"};
static const lean_object* lp_mathlib_reportAdaptationNote___closed__12 = (const lean_object*)&lp_mathlib_reportAdaptationNote___closed__12_value;
static lean_once_cell_t lp_mathlib_reportAdaptationNote___closed__13_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_reportAdaptationNote___closed__13;
static lean_once_cell_t lp_mathlib_reportAdaptationNote___closed__14_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_reportAdaptationNote___closed__14;
static const lean_string_object lp_mathlib_reportAdaptationNote___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib_reportAdaptationNote___closed__15 = (const lean_object*)&lp_mathlib_reportAdaptationNote___closed__15_value;
static const lean_ctor_object lp_mathlib_reportAdaptationNote___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_reportAdaptationNote___closed__15_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib_reportAdaptationNote___closed__16 = (const lean_object*)&lp_mathlib_reportAdaptationNote___closed__16_value;
static const lean_string_object lp_mathlib_reportAdaptationNote___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "Try this:"};
static const lean_object* lp_mathlib_reportAdaptationNote___closed__17 = (const lean_object*)&lp_mathlib_reportAdaptationNote___closed__17_value;
static const lean_ctor_object lp_mathlib_reportAdaptationNote___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Lean_addTrace___at___00reportAdaptationNote_spec__0___closed__1_value),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_reportAdaptationNote___closed__18 = (const lean_object*)&lp_mathlib_reportAdaptationNote___closed__18_value;
LEAN_EXPORT lean_object* lp_mathlib_reportAdaptationNote(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_reportAdaptationNote___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_adaptationNoteCmd___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "adaptationNoteCmd"};
static const lean_object* lp_mathlib_adaptationNoteCmd___closed__0 = (const lean_object*)&lp_mathlib_adaptationNoteCmd___closed__0_value;
static const lean_ctor_object lp_mathlib_adaptationNoteCmd___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_adaptationNoteCmd___closed__0_value),LEAN_SCALAR_PTR_LITERAL(69, 115, 184, 16, 187, 82, 203, 160)}};
static const lean_object* lp_mathlib_adaptationNoteCmd___closed__1 = (const lean_object*)&lp_mathlib_adaptationNoteCmd___closed__1_value;
static const lean_string_object lp_mathlib_adaptationNoteCmd___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_adaptationNoteCmd___closed__2 = (const lean_object*)&lp_mathlib_adaptationNoteCmd___closed__2_value;
static const lean_ctor_object lp_mathlib_adaptationNoteCmd___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_adaptationNoteCmd___closed__2_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_adaptationNoteCmd___closed__3 = (const lean_object*)&lp_mathlib_adaptationNoteCmd___closed__3_value;
static const lean_string_object lp_mathlib_adaptationNoteCmd___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "#adaptation_note "};
static const lean_object* lp_mathlib_adaptationNoteCmd___closed__4 = (const lean_object*)&lp_mathlib_adaptationNoteCmd___closed__4_value;
static const lean_ctor_object lp_mathlib_adaptationNoteCmd___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_adaptationNoteCmd___closed__4_value)}};
static const lean_object* lp_mathlib_adaptationNoteCmd___closed__5 = (const lean_object*)&lp_mathlib_adaptationNoteCmd___closed__5_value;
static const lean_string_object lp_mathlib_adaptationNoteCmd___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "optional"};
static const lean_object* lp_mathlib_adaptationNoteCmd___closed__6 = (const lean_object*)&lp_mathlib_adaptationNoteCmd___closed__6_value;
static const lean_ctor_object lp_mathlib_adaptationNoteCmd___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_adaptationNoteCmd___closed__6_value),LEAN_SCALAR_PTR_LITERAL(233, 141, 154, 50, 143, 135, 42, 252)}};
static const lean_object* lp_mathlib_adaptationNoteCmd___closed__7 = (const lean_object*)&lp_mathlib_adaptationNoteCmd___closed__7_value;
static const lean_ctor_object lp_mathlib_adaptationNoteCmd___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_reportAdaptationNote___closed__8_value),LEAN_SCALAR_PTR_LITERAL(229, 56, 215, 222, 243, 187, 251, 54)}};
static const lean_object* lp_mathlib_adaptationNoteCmd___closed__8 = (const lean_object*)&lp_mathlib_adaptationNoteCmd___closed__8_value;
static const lean_ctor_object lp_mathlib_adaptationNoteCmd___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_adaptationNoteCmd___closed__8_value)}};
static const lean_object* lp_mathlib_adaptationNoteCmd___closed__9 = (const lean_object*)&lp_mathlib_adaptationNoteCmd___closed__9_value;
static const lean_ctor_object lp_mathlib_adaptationNoteCmd___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_adaptationNoteCmd___closed__7_value),((lean_object*)&lp_mathlib_adaptationNoteCmd___closed__9_value)}};
static const lean_object* lp_mathlib_adaptationNoteCmd___closed__10 = (const lean_object*)&lp_mathlib_adaptationNoteCmd___closed__10_value;
static const lean_ctor_object lp_mathlib_adaptationNoteCmd___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_adaptationNoteCmd___closed__3_value),((lean_object*)&lp_mathlib_adaptationNoteCmd___closed__5_value),((lean_object*)&lp_mathlib_adaptationNoteCmd___closed__10_value)}};
static const lean_object* lp_mathlib_adaptationNoteCmd___closed__11 = (const lean_object*)&lp_mathlib_adaptationNoteCmd___closed__11_value;
static const lean_ctor_object lp_mathlib_adaptationNoteCmd___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_adaptationNoteCmd___closed__1_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_mathlib_adaptationNoteCmd___closed__11_value)}};
static const lean_object* lp_mathlib_adaptationNoteCmd___closed__12 = (const lean_object*)&lp_mathlib_adaptationNoteCmd___closed__12_value;
LEAN_EXPORT const lean_object* lp_mathlib_adaptationNoteCmd = (const lean_object*)&lp_mathlib_adaptationNoteCmd___closed__12_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__AdaptationNote______elabRules__adaptationNoteCmd__1_spec__0___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__AdaptationNote______elabRules__adaptationNoteCmd__1_spec__0___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__AdaptationNote______elabRules__adaptationNoteCmd__1_spec__0___redArg();
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__AdaptationNote______elabRules__adaptationNoteCmd__1_spec__0___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__AdaptationNote______elabRules__adaptationNoteCmd__1_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__AdaptationNote______elabRules__adaptationNoteCmd__1_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___aux__Mathlib__Tactic__AdaptationNote______elabRules__adaptationNoteCmd__1___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "tactic"};
static const lean_object* lp_mathlib___aux__Mathlib__Tactic__AdaptationNote______elabRules__adaptationNoteCmd__1___lam__0___closed__0 = (const lean_object*)&lp_mathlib___aux__Mathlib__Tactic__AdaptationNote______elabRules__adaptationNoteCmd__1___lam__0___closed__0_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Tactic__AdaptationNote______elabRules__adaptationNoteCmd__1___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Tactic__AdaptationNote______elabRules__adaptationNoteCmd__1___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(99, 76, 33, 121, 85, 143, 17, 224)}};
static const lean_object* lp_mathlib___aux__Mathlib__Tactic__AdaptationNote______elabRules__adaptationNoteCmd__1___lam__0___closed__1 = (const lean_object*)&lp_mathlib___aux__Mathlib__Tactic__AdaptationNote______elabRules__adaptationNoteCmd__1___lam__0___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Tactic__AdaptationNote______elabRules__adaptationNoteCmd__1___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Tactic__AdaptationNote______elabRules__adaptationNoteCmd__1___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Tactic__AdaptationNote______elabRules__adaptationNoteCmd__1___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib___aux__Mathlib__Tactic__AdaptationNote______elabRules__adaptationNoteCmd__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___aux__Mathlib__Tactic__AdaptationNote______elabRules__adaptationNoteCmd__1___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___aux__Mathlib__Tactic__AdaptationNote______elabRules__adaptationNoteCmd__1___closed__0 = (const lean_object*)&lp_mathlib___aux__Mathlib__Tactic__AdaptationNote______elabRules__adaptationNoteCmd__1___closed__0_value;
static const lean_closure_object lp_mathlib___aux__Mathlib__Tactic__AdaptationNote______elabRules__adaptationNoteCmd__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___aux__Mathlib__Tactic__AdaptationNote______elabRules__adaptationNoteCmd__1___lam__1___boxed, .m_arity = 8, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Tactic__AdaptationNote______elabRules__adaptationNoteCmd__1___closed__0_value)} };
static const lean_object* lp_mathlib___aux__Mathlib__Tactic__AdaptationNote______elabRules__adaptationNoteCmd__1___closed__1 = (const lean_object*)&lp_mathlib___aux__Mathlib__Tactic__AdaptationNote______elabRules__adaptationNoteCmd__1___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Tactic__AdaptationNote______elabRules__adaptationNoteCmd__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Tactic__AdaptationNote______elabRules__adaptationNoteCmd__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_tactic_x23adaptation__note___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 24, .m_capacity = 24, .m_length = 23, .m_data = "tactic#adaptation_note_"};
static const lean_object* lp_mathlib_tactic_x23adaptation__note___00__closed__0 = (const lean_object*)&lp_mathlib_tactic_x23adaptation__note___00__closed__0_value;
static const lean_ctor_object lp_mathlib_tactic_x23adaptation__note___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_tactic_x23adaptation__note___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(37, 97, 231, 128, 245, 117, 181, 171)}};
static const lean_object* lp_mathlib_tactic_x23adaptation__note___00__closed__1 = (const lean_object*)&lp_mathlib_tactic_x23adaptation__note___00__closed__1_value;
static const lean_ctor_object lp_mathlib_tactic_x23adaptation__note___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib_adaptationNoteCmd___closed__4_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_tactic_x23adaptation__note___00__closed__2 = (const lean_object*)&lp_mathlib_tactic_x23adaptation__note___00__closed__2_value;
static const lean_ctor_object lp_mathlib_tactic_x23adaptation__note___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_adaptationNoteCmd___closed__3_value),((lean_object*)&lp_mathlib_tactic_x23adaptation__note___00__closed__2_value),((lean_object*)&lp_mathlib_adaptationNoteCmd___closed__10_value)}};
static const lean_object* lp_mathlib_tactic_x23adaptation__note___00__closed__3 = (const lean_object*)&lp_mathlib_tactic_x23adaptation__note___00__closed__3_value;
static const lean_ctor_object lp_mathlib_tactic_x23adaptation__note___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_tactic_x23adaptation__note___00__closed__1_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_mathlib_tactic_x23adaptation__note___00__closed__3_value)}};
static const lean_object* lp_mathlib_tactic_x23adaptation__note___00__closed__4 = (const lean_object*)&lp_mathlib_tactic_x23adaptation__note___00__closed__4_value;
LEAN_EXPORT const lean_object* lp_mathlib_tactic_x23adaptation__note__ = (const lean_object*)&lp_mathlib_tactic_x23adaptation__note___00__closed__4_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__AdaptationNote______elabRules__tactic_x23adaptation__note____1_spec__0___redArg();
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__AdaptationNote______elabRules__tactic_x23adaptation__note____1_spec__0___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__AdaptationNote______elabRules__tactic_x23adaptation__note____1_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__AdaptationNote______elabRules__tactic_x23adaptation__note____1_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Tactic__AdaptationNote______elabRules__tactic_x23adaptation__note____1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Tactic__AdaptationNote______elabRules__tactic_x23adaptation__note____1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_adaptationNoteTermStx___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 22, .m_capacity = 22, .m_length = 21, .m_data = "adaptationNoteTermStx"};
static const lean_object* lp_mathlib_adaptationNoteTermStx___closed__0 = (const lean_object*)&lp_mathlib_adaptationNoteTermStx___closed__0_value;
static const lean_ctor_object lp_mathlib_adaptationNoteTermStx___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_adaptationNoteTermStx___closed__0_value),LEAN_SCALAR_PTR_LITERAL(18, 105, 211, 100, 119, 110, 131, 213)}};
static const lean_object* lp_mathlib_adaptationNoteTermStx___closed__1 = (const lean_object*)&lp_mathlib_adaptationNoteTermStx___closed__1_value;
static const lean_string_object lp_mathlib_adaptationNoteTermStx___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "term"};
static const lean_object* lp_mathlib_adaptationNoteTermStx___closed__2 = (const lean_object*)&lp_mathlib_adaptationNoteTermStx___closed__2_value;
static const lean_ctor_object lp_mathlib_adaptationNoteTermStx___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_adaptationNoteTermStx___closed__2_value),LEAN_SCALAR_PTR_LITERAL(187, 230, 181, 162, 253, 146, 122, 119)}};
static const lean_object* lp_mathlib_adaptationNoteTermStx___closed__3 = (const lean_object*)&lp_mathlib_adaptationNoteTermStx___closed__3_value;
static const lean_ctor_object lp_mathlib_adaptationNoteTermStx___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_adaptationNoteTermStx___closed__3_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_adaptationNoteTermStx___closed__4 = (const lean_object*)&lp_mathlib_adaptationNoteTermStx___closed__4_value;
static const lean_ctor_object lp_mathlib_adaptationNoteTermStx___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_adaptationNoteCmd___closed__3_value),((lean_object*)&lp_mathlib_adaptationNoteCmd___closed__11_value),((lean_object*)&lp_mathlib_adaptationNoteTermStx___closed__4_value)}};
static const lean_object* lp_mathlib_adaptationNoteTermStx___closed__5 = (const lean_object*)&lp_mathlib_adaptationNoteTermStx___closed__5_value;
static const lean_ctor_object lp_mathlib_adaptationNoteTermStx___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_adaptationNoteTermStx___closed__1_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_mathlib_adaptationNoteTermStx___closed__5_value)}};
static const lean_object* lp_mathlib_adaptationNoteTermStx___closed__6 = (const lean_object*)&lp_mathlib_adaptationNoteTermStx___closed__6_value;
LEAN_EXPORT const lean_object* lp_mathlib_adaptationNoteTermStx = (const lean_object*)&lp_mathlib_adaptationNoteTermStx___closed__6_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00adaptationNoteTermElab_spec__0___redArg();
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00adaptationNoteTermElab_spec__0___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00adaptationNoteTermElab_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00adaptationNoteTermElab_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_adaptationNoteTermElab___lam__0(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_adaptationNoteTermElab___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_adaptationNoteTermElab___lam__0, .m_arity = 2, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_mathlib_adaptationNoteTermStx___closed__3_value)} };
static const lean_object* lp_mathlib_adaptationNoteTermElab___closed__0 = (const lean_object*)&lp_mathlib_adaptationNoteTermElab___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_adaptationNoteTermElab(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_adaptationNoteTermElab___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_AdaptationNote_0__initFn_00___x40_Mathlib_Tactic_AdaptationNote_1330411147____hygCtx___hyg_2_(){
_start:
{
lean_object* v___x_55_; uint8_t v___x_56_; lean_object* v___x_57_; lean_object* v___x_58_; 
v___x_55_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_AdaptationNote_0__initFn___closed__1_00___x40_Mathlib_Tactic_AdaptationNote_1330411147____hygCtx___hyg_2_));
v___x_56_ = 0;
v___x_57_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_AdaptationNote_0__initFn___closed__23_00___x40_Mathlib_Tactic_AdaptationNote_1330411147____hygCtx___hyg_2_));
v___x_58_ = l_Lean_registerTraceClass(v___x_55_, v___x_56_, v___x_57_);
return v___x_58_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_AdaptationNote_0__initFn_00___x40_Mathlib_Tactic_AdaptationNote_1330411147____hygCtx___hyg_2____boxed(lean_object* v_a_59_){
_start:
{
lean_object* v_res_60_; 
v_res_60_ = lp_mathlib___private_Mathlib_Tactic_AdaptationNote_0__initFn_00___x40_Mathlib_Tactic_AdaptationNote_1330411147____hygCtx___hyg_2_();
return v_res_60_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_addTrace___at___00reportAdaptationNote_spec__0_spec__0(lean_object* v_msgData_61_, lean_object* v___y_62_, lean_object* v___y_63_, lean_object* v___y_64_, lean_object* v___y_65_){
_start:
{
lean_object* v___x_67_; lean_object* v_env_68_; lean_object* v___x_69_; lean_object* v_mctx_70_; lean_object* v_lctx_71_; lean_object* v_options_72_; lean_object* v___x_73_; lean_object* v___x_74_; lean_object* v___x_75_; 
v___x_67_ = lean_st_ref_get(v___y_65_);
v_env_68_ = lean_ctor_get(v___x_67_, 0);
lean_inc_ref(v_env_68_);
lean_dec(v___x_67_);
v___x_69_ = lean_st_ref_get(v___y_63_);
v_mctx_70_ = lean_ctor_get(v___x_69_, 0);
lean_inc_ref(v_mctx_70_);
lean_dec(v___x_69_);
v_lctx_71_ = lean_ctor_get(v___y_62_, 2);
v_options_72_ = lean_ctor_get(v___y_64_, 2);
lean_inc_ref(v_options_72_);
lean_inc_ref(v_lctx_71_);
v___x_73_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_73_, 0, v_env_68_);
lean_ctor_set(v___x_73_, 1, v_mctx_70_);
lean_ctor_set(v___x_73_, 2, v_lctx_71_);
lean_ctor_set(v___x_73_, 3, v_options_72_);
v___x_74_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_74_, 0, v___x_73_);
lean_ctor_set(v___x_74_, 1, v_msgData_61_);
v___x_75_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_75_, 0, v___x_74_);
return v___x_75_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_addTrace___at___00reportAdaptationNote_spec__0_spec__0___boxed(lean_object* v_msgData_76_, lean_object* v___y_77_, lean_object* v___y_78_, lean_object* v___y_79_, lean_object* v___y_80_, lean_object* v___y_81_){
_start:
{
lean_object* v_res_82_; 
v_res_82_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_addTrace___at___00reportAdaptationNote_spec__0_spec__0(v_msgData_76_, v___y_77_, v___y_78_, v___y_79_, v___y_80_);
lean_dec(v___y_80_);
lean_dec_ref(v___y_79_);
lean_dec(v___y_78_);
lean_dec_ref(v___y_77_);
return v_res_82_;
}
}
static double _init_lp_mathlib_Lean_addTrace___at___00reportAdaptationNote_spec__0___closed__0(void){
_start:
{
lean_object* v___x_83_; double v___x_84_; 
v___x_83_ = lean_unsigned_to_nat(0u);
v___x_84_ = lean_float_of_nat(v___x_83_);
return v___x_84_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00reportAdaptationNote_spec__0(lean_object* v_cls_88_, lean_object* v_msg_89_, lean_object* v___y_90_, lean_object* v___y_91_, lean_object* v___y_92_, lean_object* v___y_93_){
_start:
{
lean_object* v_ref_95_; lean_object* v___x_96_; lean_object* v_a_97_; lean_object* v___x_99_; uint8_t v_isShared_100_; uint8_t v_isSharedCheck_141_; 
v_ref_95_ = lean_ctor_get(v___y_92_, 5);
v___x_96_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_addTrace___at___00reportAdaptationNote_spec__0_spec__0(v_msg_89_, v___y_90_, v___y_91_, v___y_92_, v___y_93_);
v_a_97_ = lean_ctor_get(v___x_96_, 0);
v_isSharedCheck_141_ = !lean_is_exclusive(v___x_96_);
if (v_isSharedCheck_141_ == 0)
{
v___x_99_ = v___x_96_;
v_isShared_100_ = v_isSharedCheck_141_;
goto v_resetjp_98_;
}
else
{
lean_inc(v_a_97_);
lean_dec(v___x_96_);
v___x_99_ = lean_box(0);
v_isShared_100_ = v_isSharedCheck_141_;
goto v_resetjp_98_;
}
v_resetjp_98_:
{
lean_object* v___x_101_; lean_object* v_traceState_102_; lean_object* v_env_103_; lean_object* v_nextMacroScope_104_; lean_object* v_ngen_105_; lean_object* v_auxDeclNGen_106_; lean_object* v_cache_107_; lean_object* v_messages_108_; lean_object* v_infoState_109_; lean_object* v_snapshotTasks_110_; lean_object* v___x_112_; uint8_t v_isShared_113_; uint8_t v_isSharedCheck_140_; 
v___x_101_ = lean_st_ref_take(v___y_93_);
v_traceState_102_ = lean_ctor_get(v___x_101_, 4);
v_env_103_ = lean_ctor_get(v___x_101_, 0);
v_nextMacroScope_104_ = lean_ctor_get(v___x_101_, 1);
v_ngen_105_ = lean_ctor_get(v___x_101_, 2);
v_auxDeclNGen_106_ = lean_ctor_get(v___x_101_, 3);
v_cache_107_ = lean_ctor_get(v___x_101_, 5);
v_messages_108_ = lean_ctor_get(v___x_101_, 6);
v_infoState_109_ = lean_ctor_get(v___x_101_, 7);
v_snapshotTasks_110_ = lean_ctor_get(v___x_101_, 8);
v_isSharedCheck_140_ = !lean_is_exclusive(v___x_101_);
if (v_isSharedCheck_140_ == 0)
{
v___x_112_ = v___x_101_;
v_isShared_113_ = v_isSharedCheck_140_;
goto v_resetjp_111_;
}
else
{
lean_inc(v_snapshotTasks_110_);
lean_inc(v_infoState_109_);
lean_inc(v_messages_108_);
lean_inc(v_cache_107_);
lean_inc(v_traceState_102_);
lean_inc(v_auxDeclNGen_106_);
lean_inc(v_ngen_105_);
lean_inc(v_nextMacroScope_104_);
lean_inc(v_env_103_);
lean_dec(v___x_101_);
v___x_112_ = lean_box(0);
v_isShared_113_ = v_isSharedCheck_140_;
goto v_resetjp_111_;
}
v_resetjp_111_:
{
uint64_t v_tid_114_; lean_object* v_traces_115_; lean_object* v___x_117_; uint8_t v_isShared_118_; uint8_t v_isSharedCheck_139_; 
v_tid_114_ = lean_ctor_get_uint64(v_traceState_102_, sizeof(void*)*1);
v_traces_115_ = lean_ctor_get(v_traceState_102_, 0);
v_isSharedCheck_139_ = !lean_is_exclusive(v_traceState_102_);
if (v_isSharedCheck_139_ == 0)
{
v___x_117_ = v_traceState_102_;
v_isShared_118_ = v_isSharedCheck_139_;
goto v_resetjp_116_;
}
else
{
lean_inc(v_traces_115_);
lean_dec(v_traceState_102_);
v___x_117_ = lean_box(0);
v_isShared_118_ = v_isSharedCheck_139_;
goto v_resetjp_116_;
}
v_resetjp_116_:
{
lean_object* v___x_119_; double v___x_120_; uint8_t v___x_121_; lean_object* v___x_122_; lean_object* v___x_123_; lean_object* v___x_124_; lean_object* v___x_125_; lean_object* v___x_126_; lean_object* v___x_127_; lean_object* v___x_129_; 
v___x_119_ = lean_box(0);
v___x_120_ = lean_float_once(&lp_mathlib_Lean_addTrace___at___00reportAdaptationNote_spec__0___closed__0, &lp_mathlib_Lean_addTrace___at___00reportAdaptationNote_spec__0___closed__0_once, _init_lp_mathlib_Lean_addTrace___at___00reportAdaptationNote_spec__0___closed__0);
v___x_121_ = 0;
v___x_122_ = ((lean_object*)(lp_mathlib_Lean_addTrace___at___00reportAdaptationNote_spec__0___closed__1));
v___x_123_ = lean_alloc_ctor(0, 3, 17);
lean_ctor_set(v___x_123_, 0, v_cls_88_);
lean_ctor_set(v___x_123_, 1, v___x_119_);
lean_ctor_set(v___x_123_, 2, v___x_122_);
lean_ctor_set_float(v___x_123_, sizeof(void*)*3, v___x_120_);
lean_ctor_set_float(v___x_123_, sizeof(void*)*3 + 8, v___x_120_);
lean_ctor_set_uint8(v___x_123_, sizeof(void*)*3 + 16, v___x_121_);
v___x_124_ = ((lean_object*)(lp_mathlib_Lean_addTrace___at___00reportAdaptationNote_spec__0___closed__2));
v___x_125_ = lean_alloc_ctor(9, 3, 0);
lean_ctor_set(v___x_125_, 0, v___x_123_);
lean_ctor_set(v___x_125_, 1, v_a_97_);
lean_ctor_set(v___x_125_, 2, v___x_124_);
lean_inc(v_ref_95_);
v___x_126_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_126_, 0, v_ref_95_);
lean_ctor_set(v___x_126_, 1, v___x_125_);
v___x_127_ = l_Lean_PersistentArray_push___redArg(v_traces_115_, v___x_126_);
if (v_isShared_118_ == 0)
{
lean_ctor_set(v___x_117_, 0, v___x_127_);
v___x_129_ = v___x_117_;
goto v_reusejp_128_;
}
else
{
lean_object* v_reuseFailAlloc_138_; 
v_reuseFailAlloc_138_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v_reuseFailAlloc_138_, 0, v___x_127_);
lean_ctor_set_uint64(v_reuseFailAlloc_138_, sizeof(void*)*1, v_tid_114_);
v___x_129_ = v_reuseFailAlloc_138_;
goto v_reusejp_128_;
}
v_reusejp_128_:
{
lean_object* v___x_131_; 
if (v_isShared_113_ == 0)
{
lean_ctor_set(v___x_112_, 4, v___x_129_);
v___x_131_ = v___x_112_;
goto v_reusejp_130_;
}
else
{
lean_object* v_reuseFailAlloc_137_; 
v_reuseFailAlloc_137_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_137_, 0, v_env_103_);
lean_ctor_set(v_reuseFailAlloc_137_, 1, v_nextMacroScope_104_);
lean_ctor_set(v_reuseFailAlloc_137_, 2, v_ngen_105_);
lean_ctor_set(v_reuseFailAlloc_137_, 3, v_auxDeclNGen_106_);
lean_ctor_set(v_reuseFailAlloc_137_, 4, v___x_129_);
lean_ctor_set(v_reuseFailAlloc_137_, 5, v_cache_107_);
lean_ctor_set(v_reuseFailAlloc_137_, 6, v_messages_108_);
lean_ctor_set(v_reuseFailAlloc_137_, 7, v_infoState_109_);
lean_ctor_set(v_reuseFailAlloc_137_, 8, v_snapshotTasks_110_);
v___x_131_ = v_reuseFailAlloc_137_;
goto v_reusejp_130_;
}
v_reusejp_130_:
{
lean_object* v___x_132_; lean_object* v___x_133_; lean_object* v___x_135_; 
v___x_132_ = lean_st_ref_set(v___y_93_, v___x_131_);
v___x_133_ = lean_box(0);
if (v_isShared_100_ == 0)
{
lean_ctor_set(v___x_99_, 0, v___x_133_);
v___x_135_ = v___x_99_;
goto v_reusejp_134_;
}
else
{
lean_object* v_reuseFailAlloc_136_; 
v_reuseFailAlloc_136_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_136_, 0, v___x_133_);
v___x_135_ = v_reuseFailAlloc_136_;
goto v_reusejp_134_;
}
v_reusejp_134_:
{
return v___x_135_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00reportAdaptationNote_spec__0___boxed(lean_object* v_cls_142_, lean_object* v_msg_143_, lean_object* v___y_144_, lean_object* v___y_145_, lean_object* v___y_146_, lean_object* v___y_147_, lean_object* v___y_148_){
_start:
{
lean_object* v_res_149_; 
v_res_149_ = lp_mathlib_Lean_addTrace___at___00reportAdaptationNote_spec__0(v_cls_142_, v_msg_143_, v___y_144_, v___y_145_, v___y_146_, v___y_147_);
lean_dec(v___y_147_);
lean_dec_ref(v___y_146_);
lean_dec(v___y_145_);
lean_dec_ref(v___y_144_);
return v_res_149_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logError___at___00reportAdaptationNote_spec__1_spec__2_spec__3___lam__0(uint8_t v___y_157_, uint8_t v_suppressElabErrors_158_, lean_object* v_x_159_){
_start:
{
if (lean_obj_tag(v_x_159_) == 1)
{
lean_object* v_pre_160_; 
v_pre_160_ = lean_ctor_get(v_x_159_, 0);
switch(lean_obj_tag(v_pre_160_))
{
case 1:
{
lean_object* v_pre_161_; 
v_pre_161_ = lean_ctor_get(v_pre_160_, 0);
switch(lean_obj_tag(v_pre_161_))
{
case 0:
{
lean_object* v_str_162_; lean_object* v_str_163_; lean_object* v___x_164_; uint8_t v___x_165_; 
v_str_162_ = lean_ctor_get(v_x_159_, 1);
v_str_163_ = lean_ctor_get(v_pre_160_, 1);
v___x_164_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logError___at___00reportAdaptationNote_spec__1_spec__2_spec__3___lam__0___closed__0));
v___x_165_ = lean_string_dec_eq(v_str_163_, v___x_164_);
if (v___x_165_ == 0)
{
lean_object* v___x_166_; uint8_t v___x_167_; 
v___x_166_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_AdaptationNote_0__initFn___closed__6_00___x40_Mathlib_Tactic_AdaptationNote_1330411147____hygCtx___hyg_2_));
v___x_167_ = lean_string_dec_eq(v_str_163_, v___x_166_);
if (v___x_167_ == 0)
{
return v___y_157_;
}
else
{
lean_object* v___x_168_; uint8_t v___x_169_; 
v___x_168_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logError___at___00reportAdaptationNote_spec__1_spec__2_spec__3___lam__0___closed__1));
v___x_169_ = lean_string_dec_eq(v_str_162_, v___x_168_);
if (v___x_169_ == 0)
{
return v___y_157_;
}
else
{
return v_suppressElabErrors_158_;
}
}
}
else
{
lean_object* v___x_170_; uint8_t v___x_171_; 
v___x_170_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logError___at___00reportAdaptationNote_spec__1_spec__2_spec__3___lam__0___closed__2));
v___x_171_ = lean_string_dec_eq(v_str_162_, v___x_170_);
if (v___x_171_ == 0)
{
return v___y_157_;
}
else
{
return v_suppressElabErrors_158_;
}
}
}
case 1:
{
lean_object* v_pre_172_; 
v_pre_172_ = lean_ctor_get(v_pre_161_, 0);
if (lean_obj_tag(v_pre_172_) == 0)
{
lean_object* v_str_173_; lean_object* v_str_174_; lean_object* v_str_175_; lean_object* v___x_176_; uint8_t v___x_177_; 
v_str_173_ = lean_ctor_get(v_x_159_, 1);
v_str_174_ = lean_ctor_get(v_pre_160_, 1);
v_str_175_ = lean_ctor_get(v_pre_161_, 1);
v___x_176_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logError___at___00reportAdaptationNote_spec__1_spec__2_spec__3___lam__0___closed__3));
v___x_177_ = lean_string_dec_eq(v_str_175_, v___x_176_);
if (v___x_177_ == 0)
{
return v___y_157_;
}
else
{
lean_object* v___x_178_; uint8_t v___x_179_; 
v___x_178_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logError___at___00reportAdaptationNote_spec__1_spec__2_spec__3___lam__0___closed__4));
v___x_179_ = lean_string_dec_eq(v_str_174_, v___x_178_);
if (v___x_179_ == 0)
{
return v___y_157_;
}
else
{
lean_object* v___x_180_; uint8_t v___x_181_; 
v___x_180_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logError___at___00reportAdaptationNote_spec__1_spec__2_spec__3___lam__0___closed__5));
v___x_181_ = lean_string_dec_eq(v_str_173_, v___x_180_);
if (v___x_181_ == 0)
{
return v___y_157_;
}
else
{
return v_suppressElabErrors_158_;
}
}
}
}
else
{
return v___y_157_;
}
}
default: 
{
return v___y_157_;
}
}
}
case 0:
{
lean_object* v_str_182_; lean_object* v___x_183_; uint8_t v___x_184_; 
v_str_182_ = lean_ctor_get(v_x_159_, 1);
v___x_183_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logError___at___00reportAdaptationNote_spec__1_spec__2_spec__3___lam__0___closed__6));
v___x_184_ = lean_string_dec_eq(v_str_182_, v___x_183_);
if (v___x_184_ == 0)
{
return v___y_157_;
}
else
{
return v_suppressElabErrors_158_;
}
}
default: 
{
return v___y_157_;
}
}
}
else
{
return v___y_157_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logError___at___00reportAdaptationNote_spec__1_spec__2_spec__3___lam__0___boxed(lean_object* v___y_185_, lean_object* v_suppressElabErrors_186_, lean_object* v_x_187_){
_start:
{
uint8_t v___y_5690__boxed_188_; uint8_t v_suppressElabErrors_boxed_189_; uint8_t v_res_190_; lean_object* v_r_191_; 
v___y_5690__boxed_188_ = lean_unbox(v___y_185_);
v_suppressElabErrors_boxed_189_ = lean_unbox(v_suppressElabErrors_186_);
v_res_190_ = lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logError___at___00reportAdaptationNote_spec__1_spec__2_spec__3___lam__0(v___y_5690__boxed_188_, v_suppressElabErrors_boxed_189_, v_x_187_);
lean_dec(v_x_187_);
v_r_191_ = lean_box(v_res_190_);
return v_r_191_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_log___at___00Lean_logError___at___00reportAdaptationNote_spec__1_spec__2_spec__3_spec__4(lean_object* v_opts_192_, lean_object* v_opt_193_){
_start:
{
lean_object* v_name_194_; lean_object* v_defValue_195_; lean_object* v_map_196_; lean_object* v___x_197_; 
v_name_194_ = lean_ctor_get(v_opt_193_, 0);
v_defValue_195_ = lean_ctor_get(v_opt_193_, 1);
v_map_196_ = lean_ctor_get(v_opts_192_, 0);
v___x_197_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_196_, v_name_194_);
if (lean_obj_tag(v___x_197_) == 0)
{
uint8_t v___x_198_; 
v___x_198_ = lean_unbox(v_defValue_195_);
return v___x_198_;
}
else
{
lean_object* v_val_199_; 
v_val_199_ = lean_ctor_get(v___x_197_, 0);
lean_inc(v_val_199_);
lean_dec_ref_known(v___x_197_, 1);
if (lean_obj_tag(v_val_199_) == 1)
{
uint8_t v_v_200_; 
v_v_200_ = lean_ctor_get_uint8(v_val_199_, 0);
lean_dec_ref_known(v_val_199_, 0);
return v_v_200_;
}
else
{
uint8_t v___x_201_; 
lean_dec(v_val_199_);
v___x_201_ = lean_unbox(v_defValue_195_);
return v___x_201_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_log___at___00Lean_logError___at___00reportAdaptationNote_spec__1_spec__2_spec__3_spec__4___boxed(lean_object* v_opts_202_, lean_object* v_opt_203_){
_start:
{
uint8_t v_res_204_; lean_object* v_r_205_; 
v_res_204_ = lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_log___at___00Lean_logError___at___00reportAdaptationNote_spec__1_spec__2_spec__3_spec__4(v_opts_202_, v_opt_203_);
lean_dec_ref(v_opt_203_);
lean_dec_ref(v_opts_202_);
v_r_205_ = lean_box(v_res_204_);
return v_r_205_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logError___at___00reportAdaptationNote_spec__1_spec__2_spec__3(lean_object* v_ref_206_, lean_object* v_msgData_207_, uint8_t v_severity_208_, uint8_t v_isSilent_209_, lean_object* v___y_210_, lean_object* v___y_211_, lean_object* v___y_212_, lean_object* v___y_213_){
_start:
{
uint8_t v___y_216_; lean_object* v___y_217_; uint8_t v___y_218_; lean_object* v___y_219_; lean_object* v___y_220_; lean_object* v___y_221_; lean_object* v___y_222_; lean_object* v___y_223_; lean_object* v___y_224_; lean_object* v___y_252_; uint8_t v___y_253_; lean_object* v___y_254_; uint8_t v___y_255_; uint8_t v___y_256_; lean_object* v___y_257_; lean_object* v___y_258_; lean_object* v___y_259_; lean_object* v___y_277_; uint8_t v___y_278_; uint8_t v___y_279_; lean_object* v___y_280_; uint8_t v___y_281_; lean_object* v___y_282_; lean_object* v___y_283_; lean_object* v___y_284_; lean_object* v___y_288_; uint8_t v___y_289_; uint8_t v___y_290_; lean_object* v___y_291_; lean_object* v___y_292_; lean_object* v___y_293_; uint8_t v___y_294_; uint8_t v___x_299_; uint8_t v___y_301_; lean_object* v___y_302_; lean_object* v___y_303_; lean_object* v___y_304_; lean_object* v___y_305_; uint8_t v___y_306_; uint8_t v___y_307_; uint8_t v___y_309_; uint8_t v___x_324_; 
v___x_299_ = 2;
v___x_324_ = l_Lean_instBEqMessageSeverity_beq(v_severity_208_, v___x_299_);
if (v___x_324_ == 0)
{
v___y_309_ = v___x_324_;
goto v___jp_308_;
}
else
{
uint8_t v___x_325_; 
lean_inc_ref(v_msgData_207_);
v___x_325_ = l_Lean_MessageData_hasSyntheticSorry(v_msgData_207_);
v___y_309_ = v___x_325_;
goto v___jp_308_;
}
v___jp_215_:
{
lean_object* v___x_225_; lean_object* v_currNamespace_226_; lean_object* v_openDecls_227_; lean_object* v_env_228_; lean_object* v_nextMacroScope_229_; lean_object* v_ngen_230_; lean_object* v_auxDeclNGen_231_; lean_object* v_traceState_232_; lean_object* v_cache_233_; lean_object* v_messages_234_; lean_object* v_infoState_235_; lean_object* v_snapshotTasks_236_; lean_object* v___x_238_; uint8_t v_isShared_239_; uint8_t v_isSharedCheck_250_; 
v___x_225_ = lean_st_ref_take(v___y_224_);
v_currNamespace_226_ = lean_ctor_get(v___y_223_, 6);
v_openDecls_227_ = lean_ctor_get(v___y_223_, 7);
v_env_228_ = lean_ctor_get(v___x_225_, 0);
v_nextMacroScope_229_ = lean_ctor_get(v___x_225_, 1);
v_ngen_230_ = lean_ctor_get(v___x_225_, 2);
v_auxDeclNGen_231_ = lean_ctor_get(v___x_225_, 3);
v_traceState_232_ = lean_ctor_get(v___x_225_, 4);
v_cache_233_ = lean_ctor_get(v___x_225_, 5);
v_messages_234_ = lean_ctor_get(v___x_225_, 6);
v_infoState_235_ = lean_ctor_get(v___x_225_, 7);
v_snapshotTasks_236_ = lean_ctor_get(v___x_225_, 8);
v_isSharedCheck_250_ = !lean_is_exclusive(v___x_225_);
if (v_isSharedCheck_250_ == 0)
{
v___x_238_ = v___x_225_;
v_isShared_239_ = v_isSharedCheck_250_;
goto v_resetjp_237_;
}
else
{
lean_inc(v_snapshotTasks_236_);
lean_inc(v_infoState_235_);
lean_inc(v_messages_234_);
lean_inc(v_cache_233_);
lean_inc(v_traceState_232_);
lean_inc(v_auxDeclNGen_231_);
lean_inc(v_ngen_230_);
lean_inc(v_nextMacroScope_229_);
lean_inc(v_env_228_);
lean_dec(v___x_225_);
v___x_238_ = lean_box(0);
v_isShared_239_ = v_isSharedCheck_250_;
goto v_resetjp_237_;
}
v_resetjp_237_:
{
lean_object* v___x_240_; lean_object* v___x_241_; lean_object* v___x_242_; lean_object* v___x_243_; lean_object* v___x_245_; 
lean_inc(v_openDecls_227_);
lean_inc(v_currNamespace_226_);
v___x_240_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_240_, 0, v_currNamespace_226_);
lean_ctor_set(v___x_240_, 1, v_openDecls_227_);
v___x_241_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_241_, 0, v___x_240_);
lean_ctor_set(v___x_241_, 1, v___y_221_);
lean_inc_ref(v___y_219_);
lean_inc_ref(v___y_220_);
v___x_242_ = lean_alloc_ctor(0, 5, 3);
lean_ctor_set(v___x_242_, 0, v___y_220_);
lean_ctor_set(v___x_242_, 1, v___y_222_);
lean_ctor_set(v___x_242_, 2, v___y_217_);
lean_ctor_set(v___x_242_, 3, v___y_219_);
lean_ctor_set(v___x_242_, 4, v___x_241_);
lean_ctor_set_uint8(v___x_242_, sizeof(void*)*5, v___y_218_);
lean_ctor_set_uint8(v___x_242_, sizeof(void*)*5 + 1, v___y_216_);
lean_ctor_set_uint8(v___x_242_, sizeof(void*)*5 + 2, v_isSilent_209_);
v___x_243_ = l_Lean_MessageLog_add(v___x_242_, v_messages_234_);
if (v_isShared_239_ == 0)
{
lean_ctor_set(v___x_238_, 6, v___x_243_);
v___x_245_ = v___x_238_;
goto v_reusejp_244_;
}
else
{
lean_object* v_reuseFailAlloc_249_; 
v_reuseFailAlloc_249_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_249_, 0, v_env_228_);
lean_ctor_set(v_reuseFailAlloc_249_, 1, v_nextMacroScope_229_);
lean_ctor_set(v_reuseFailAlloc_249_, 2, v_ngen_230_);
lean_ctor_set(v_reuseFailAlloc_249_, 3, v_auxDeclNGen_231_);
lean_ctor_set(v_reuseFailAlloc_249_, 4, v_traceState_232_);
lean_ctor_set(v_reuseFailAlloc_249_, 5, v_cache_233_);
lean_ctor_set(v_reuseFailAlloc_249_, 6, v___x_243_);
lean_ctor_set(v_reuseFailAlloc_249_, 7, v_infoState_235_);
lean_ctor_set(v_reuseFailAlloc_249_, 8, v_snapshotTasks_236_);
v___x_245_ = v_reuseFailAlloc_249_;
goto v_reusejp_244_;
}
v_reusejp_244_:
{
lean_object* v___x_246_; lean_object* v___x_247_; lean_object* v___x_248_; 
v___x_246_ = lean_st_ref_set(v___y_224_, v___x_245_);
v___x_247_ = lean_box(0);
v___x_248_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_248_, 0, v___x_247_);
return v___x_248_;
}
}
}
v___jp_251_:
{
lean_object* v___x_260_; lean_object* v___x_261_; lean_object* v_a_262_; lean_object* v___x_264_; uint8_t v_isShared_265_; uint8_t v_isSharedCheck_275_; 
v___x_260_ = l___private_Lean_Log_0__Lean_MessageData_appendDescriptionWidgetIfNamed(v_msgData_207_);
v___x_261_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_addTrace___at___00reportAdaptationNote_spec__0_spec__0(v___x_260_, v___y_210_, v___y_211_, v___y_212_, v___y_213_);
v_a_262_ = lean_ctor_get(v___x_261_, 0);
v_isSharedCheck_275_ = !lean_is_exclusive(v___x_261_);
if (v_isSharedCheck_275_ == 0)
{
v___x_264_ = v___x_261_;
v_isShared_265_ = v_isSharedCheck_275_;
goto v_resetjp_263_;
}
else
{
lean_inc(v_a_262_);
lean_dec(v___x_261_);
v___x_264_ = lean_box(0);
v_isShared_265_ = v_isSharedCheck_275_;
goto v_resetjp_263_;
}
v_resetjp_263_:
{
lean_object* v___x_266_; lean_object* v___x_267_; lean_object* v___x_268_; lean_object* v___x_269_; 
lean_inc_ref_n(v___y_257_, 2);
v___x_266_ = l_Lean_FileMap_toPosition(v___y_257_, v___y_254_);
lean_dec(v___y_254_);
v___x_267_ = l_Lean_FileMap_toPosition(v___y_257_, v___y_259_);
lean_dec(v___y_259_);
v___x_268_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_268_, 0, v___x_267_);
v___x_269_ = ((lean_object*)(lp_mathlib_Lean_addTrace___at___00reportAdaptationNote_spec__0___closed__1));
if (v___y_253_ == 0)
{
lean_del_object(v___x_264_);
lean_dec_ref(v___y_252_);
v___y_216_ = v___y_255_;
v___y_217_ = v___x_268_;
v___y_218_ = v___y_256_;
v___y_219_ = v___x_269_;
v___y_220_ = v___y_258_;
v___y_221_ = v_a_262_;
v___y_222_ = v___x_266_;
v___y_223_ = v___y_212_;
v___y_224_ = v___y_213_;
goto v___jp_215_;
}
else
{
uint8_t v___x_270_; 
lean_inc(v_a_262_);
v___x_270_ = l_Lean_MessageData_hasTag(v___y_252_, v_a_262_);
if (v___x_270_ == 0)
{
lean_object* v___x_271_; lean_object* v___x_273_; 
lean_dec_ref_known(v___x_268_, 1);
lean_dec_ref(v___x_266_);
lean_dec(v_a_262_);
v___x_271_ = lean_box(0);
if (v_isShared_265_ == 0)
{
lean_ctor_set(v___x_264_, 0, v___x_271_);
v___x_273_ = v___x_264_;
goto v_reusejp_272_;
}
else
{
lean_object* v_reuseFailAlloc_274_; 
v_reuseFailAlloc_274_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_274_, 0, v___x_271_);
v___x_273_ = v_reuseFailAlloc_274_;
goto v_reusejp_272_;
}
v_reusejp_272_:
{
return v___x_273_;
}
}
else
{
lean_del_object(v___x_264_);
v___y_216_ = v___y_255_;
v___y_217_ = v___x_268_;
v___y_218_ = v___y_256_;
v___y_219_ = v___x_269_;
v___y_220_ = v___y_258_;
v___y_221_ = v_a_262_;
v___y_222_ = v___x_266_;
v___y_223_ = v___y_212_;
v___y_224_ = v___y_213_;
goto v___jp_215_;
}
}
}
}
v___jp_276_:
{
lean_object* v___x_285_; 
v___x_285_ = l_Lean_Syntax_getTailPos_x3f(v___y_282_, v___y_281_);
lean_dec(v___y_282_);
if (lean_obj_tag(v___x_285_) == 0)
{
lean_inc(v___y_284_);
v___y_252_ = v___y_277_;
v___y_253_ = v___y_278_;
v___y_254_ = v___y_284_;
v___y_255_ = v___y_279_;
v___y_256_ = v___y_281_;
v___y_257_ = v___y_280_;
v___y_258_ = v___y_283_;
v___y_259_ = v___y_284_;
goto v___jp_251_;
}
else
{
lean_object* v_val_286_; 
v_val_286_ = lean_ctor_get(v___x_285_, 0);
lean_inc(v_val_286_);
lean_dec_ref_known(v___x_285_, 1);
v___y_252_ = v___y_277_;
v___y_253_ = v___y_278_;
v___y_254_ = v___y_284_;
v___y_255_ = v___y_279_;
v___y_256_ = v___y_281_;
v___y_257_ = v___y_280_;
v___y_258_ = v___y_283_;
v___y_259_ = v_val_286_;
goto v___jp_251_;
}
}
v___jp_287_:
{
lean_object* v_ref_295_; lean_object* v___x_296_; 
v_ref_295_ = l_Lean_replaceRef(v_ref_206_, v___y_292_);
v___x_296_ = l_Lean_Syntax_getPos_x3f(v_ref_295_, v___y_290_);
if (lean_obj_tag(v___x_296_) == 0)
{
lean_object* v___x_297_; 
v___x_297_ = lean_unsigned_to_nat(0u);
v___y_277_ = v___y_288_;
v___y_278_ = v___y_289_;
v___y_279_ = v___y_294_;
v___y_280_ = v___y_291_;
v___y_281_ = v___y_290_;
v___y_282_ = v_ref_295_;
v___y_283_ = v___y_293_;
v___y_284_ = v___x_297_;
goto v___jp_276_;
}
else
{
lean_object* v_val_298_; 
v_val_298_ = lean_ctor_get(v___x_296_, 0);
lean_inc(v_val_298_);
lean_dec_ref_known(v___x_296_, 1);
v___y_277_ = v___y_288_;
v___y_278_ = v___y_289_;
v___y_279_ = v___y_294_;
v___y_280_ = v___y_291_;
v___y_281_ = v___y_290_;
v___y_282_ = v_ref_295_;
v___y_283_ = v___y_293_;
v___y_284_ = v_val_298_;
goto v___jp_276_;
}
}
v___jp_300_:
{
if (v___y_307_ == 0)
{
v___y_288_ = v___y_305_;
v___y_289_ = v___y_301_;
v___y_290_ = v___y_306_;
v___y_291_ = v___y_302_;
v___y_292_ = v___y_303_;
v___y_293_ = v___y_304_;
v___y_294_ = v_severity_208_;
goto v___jp_287_;
}
else
{
v___y_288_ = v___y_305_;
v___y_289_ = v___y_301_;
v___y_290_ = v___y_306_;
v___y_291_ = v___y_302_;
v___y_292_ = v___y_303_;
v___y_293_ = v___y_304_;
v___y_294_ = v___x_299_;
goto v___jp_287_;
}
}
v___jp_308_:
{
if (v___y_309_ == 0)
{
lean_object* v_fileName_310_; lean_object* v_fileMap_311_; lean_object* v_options_312_; lean_object* v_ref_313_; uint8_t v_suppressElabErrors_314_; lean_object* v___x_315_; lean_object* v___x_316_; lean_object* v___f_317_; uint8_t v___x_318_; uint8_t v___x_319_; 
v_fileName_310_ = lean_ctor_get(v___y_212_, 0);
v_fileMap_311_ = lean_ctor_get(v___y_212_, 1);
v_options_312_ = lean_ctor_get(v___y_212_, 2);
v_ref_313_ = lean_ctor_get(v___y_212_, 5);
v_suppressElabErrors_314_ = lean_ctor_get_uint8(v___y_212_, sizeof(void*)*14 + 1);
v___x_315_ = lean_box(v___y_309_);
v___x_316_ = lean_box(v_suppressElabErrors_314_);
v___f_317_ = lean_alloc_closure((void*)(lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logError___at___00reportAdaptationNote_spec__1_spec__2_spec__3___lam__0___boxed), 3, 2);
lean_closure_set(v___f_317_, 0, v___x_315_);
lean_closure_set(v___f_317_, 1, v___x_316_);
v___x_318_ = 1;
v___x_319_ = l_Lean_instBEqMessageSeverity_beq(v_severity_208_, v___x_318_);
if (v___x_319_ == 0)
{
v___y_301_ = v_suppressElabErrors_314_;
v___y_302_ = v_fileMap_311_;
v___y_303_ = v_ref_313_;
v___y_304_ = v_fileName_310_;
v___y_305_ = v___f_317_;
v___y_306_ = v___y_309_;
v___y_307_ = v___x_319_;
goto v___jp_300_;
}
else
{
lean_object* v___x_320_; uint8_t v___x_321_; 
v___x_320_ = l_Lean_warningAsError;
v___x_321_ = lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_log___at___00Lean_logError___at___00reportAdaptationNote_spec__1_spec__2_spec__3_spec__4(v_options_312_, v___x_320_);
v___y_301_ = v_suppressElabErrors_314_;
v___y_302_ = v_fileMap_311_;
v___y_303_ = v_ref_313_;
v___y_304_ = v_fileName_310_;
v___y_305_ = v___f_317_;
v___y_306_ = v___y_309_;
v___y_307_ = v___x_321_;
goto v___jp_300_;
}
}
else
{
lean_object* v___x_322_; lean_object* v___x_323_; 
lean_dec_ref(v_msgData_207_);
v___x_322_ = lean_box(0);
v___x_323_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_323_, 0, v___x_322_);
return v___x_323_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logError___at___00reportAdaptationNote_spec__1_spec__2_spec__3___boxed(lean_object* v_ref_326_, lean_object* v_msgData_327_, lean_object* v_severity_328_, lean_object* v_isSilent_329_, lean_object* v___y_330_, lean_object* v___y_331_, lean_object* v___y_332_, lean_object* v___y_333_, lean_object* v___y_334_){
_start:
{
uint8_t v_severity_boxed_335_; uint8_t v_isSilent_boxed_336_; lean_object* v_res_337_; 
v_severity_boxed_335_ = lean_unbox(v_severity_328_);
v_isSilent_boxed_336_ = lean_unbox(v_isSilent_329_);
v_res_337_ = lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logError___at___00reportAdaptationNote_spec__1_spec__2_spec__3(v_ref_326_, v_msgData_327_, v_severity_boxed_335_, v_isSilent_boxed_336_, v___y_330_, v___y_331_, v___y_332_, v___y_333_);
lean_dec(v___y_333_);
lean_dec_ref(v___y_332_);
lean_dec(v___y_331_);
lean_dec_ref(v___y_330_);
lean_dec(v_ref_326_);
return v_res_337_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_log___at___00Lean_logError___at___00reportAdaptationNote_spec__1_spec__2(lean_object* v_msgData_338_, uint8_t v_severity_339_, uint8_t v_isSilent_340_, lean_object* v___y_341_, lean_object* v___y_342_, lean_object* v___y_343_, lean_object* v___y_344_){
_start:
{
lean_object* v_ref_346_; lean_object* v___x_347_; 
v_ref_346_ = lean_ctor_get(v___y_343_, 5);
v___x_347_ = lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logError___at___00reportAdaptationNote_spec__1_spec__2_spec__3(v_ref_346_, v_msgData_338_, v_severity_339_, v_isSilent_340_, v___y_341_, v___y_342_, v___y_343_, v___y_344_);
return v___x_347_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_log___at___00Lean_logError___at___00reportAdaptationNote_spec__1_spec__2___boxed(lean_object* v_msgData_348_, lean_object* v_severity_349_, lean_object* v_isSilent_350_, lean_object* v___y_351_, lean_object* v___y_352_, lean_object* v___y_353_, lean_object* v___y_354_, lean_object* v___y_355_){
_start:
{
uint8_t v_severity_boxed_356_; uint8_t v_isSilent_boxed_357_; lean_object* v_res_358_; 
v_severity_boxed_356_ = lean_unbox(v_severity_349_);
v_isSilent_boxed_357_ = lean_unbox(v_isSilent_350_);
v_res_358_ = lp_mathlib_Lean_log___at___00Lean_logError___at___00reportAdaptationNote_spec__1_spec__2(v_msgData_348_, v_severity_boxed_356_, v_isSilent_boxed_357_, v___y_351_, v___y_352_, v___y_353_, v___y_354_);
lean_dec(v___y_354_);
lean_dec_ref(v___y_353_);
lean_dec(v___y_352_);
lean_dec_ref(v___y_351_);
return v_res_358_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logError___at___00reportAdaptationNote_spec__1(lean_object* v_msgData_359_, lean_object* v___y_360_, lean_object* v___y_361_, lean_object* v___y_362_, lean_object* v___y_363_){
_start:
{
uint8_t v___x_365_; uint8_t v___x_366_; lean_object* v___x_367_; 
v___x_365_ = 2;
v___x_366_ = 0;
v___x_367_ = lp_mathlib_Lean_log___at___00Lean_logError___at___00reportAdaptationNote_spec__1_spec__2(v_msgData_359_, v___x_365_, v___x_366_, v___y_360_, v___y_361_, v___y_362_, v___y_363_);
return v___x_367_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logError___at___00reportAdaptationNote_spec__1___boxed(lean_object* v_msgData_368_, lean_object* v___y_369_, lean_object* v___y_370_, lean_object* v___y_371_, lean_object* v___y_372_, lean_object* v___y_373_){
_start:
{
lean_object* v_res_374_; 
v_res_374_ = lp_mathlib_Lean_logError___at___00reportAdaptationNote_spec__1(v_msgData_368_, v___y_369_, v___y_370_, v___y_371_, v___y_372_);
lean_dec(v___y_372_);
lean_dec_ref(v___y_371_);
lean_dec(v___y_370_);
lean_dec_ref(v___y_369_);
return v_res_374_;
}
}
static lean_object* _init_lp_mathlib_reportAdaptationNote___closed__1(void){
_start:
{
lean_object* v___x_377_; lean_object* v___x_378_; lean_object* v___x_379_; 
v___x_377_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_AdaptationNote_0__initFn___closed__1_00___x40_Mathlib_Tactic_AdaptationNote_1330411147____hygCtx___hyg_2_));
v___x_378_ = ((lean_object*)(lp_mathlib_reportAdaptationNote___closed__0));
v___x_379_ = l_Lean_Name_append(v___x_378_, v___x_377_);
return v___x_379_;
}
}
static lean_object* _init_lp_mathlib_reportAdaptationNote___closed__4(void){
_start:
{
lean_object* v___x_383_; lean_object* v___x_384_; 
v___x_383_ = ((lean_object*)(lp_mathlib_reportAdaptationNote___closed__3));
v___x_384_ = l_Lean_MessageData_ofFormat(v___x_383_);
return v___x_384_;
}
}
static lean_object* _init_lp_mathlib_reportAdaptationNote___closed__11(void){
_start:
{
lean_object* v___x_395_; lean_object* v___x_396_; 
v___x_395_ = ((lean_object*)(lp_mathlib_reportAdaptationNote___closed__10));
v___x_396_ = l_Lean_mkAtom(v___x_395_);
return v___x_396_;
}
}
static lean_object* _init_lp_mathlib_reportAdaptationNote___closed__13(void){
_start:
{
lean_object* v___x_398_; lean_object* v___x_399_; 
v___x_398_ = ((lean_object*)(lp_mathlib_reportAdaptationNote___closed__12));
v___x_399_ = l_Lean_mkAtom(v___x_398_);
return v___x_399_;
}
}
static lean_object* _init_lp_mathlib_reportAdaptationNote___closed__14(void){
_start:
{
lean_object* v___x_400_; lean_object* v___x_401_; lean_object* v___x_402_; lean_object* v___x_403_; lean_object* v___x_404_; 
v___x_400_ = lean_obj_once(&lp_mathlib_reportAdaptationNote___closed__13, &lp_mathlib_reportAdaptationNote___closed__13_once, _init_lp_mathlib_reportAdaptationNote___closed__13);
v___x_401_ = lean_obj_once(&lp_mathlib_reportAdaptationNote___closed__11, &lp_mathlib_reportAdaptationNote___closed__11_once, _init_lp_mathlib_reportAdaptationNote___closed__11);
v___x_402_ = ((lean_object*)(lp_mathlib_reportAdaptationNote___closed__9));
v___x_403_ = lean_box(2);
v___x_404_ = l_Lean_Syntax_node2(v___x_403_, v___x_402_, v___x_401_, v___x_400_);
return v___x_404_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_reportAdaptationNote(lean_object* v_f_412_, lean_object* v_a_413_, lean_object* v_a_414_, lean_object* v_a_415_, lean_object* v_a_416_){
_start:
{
lean_object* v_options_421_; lean_object* v_ref_422_; lean_object* v_inheritedTraceOptions_423_; lean_object* v___x_424_; lean_object* v___x_425_; lean_object* v___x_426_; 
v_options_421_ = lean_ctor_get(v_a_415_, 2);
v_ref_422_ = lean_ctor_get(v_a_415_, 5);
v_inheritedTraceOptions_423_ = lean_ctor_get(v_a_415_, 13);
v___x_424_ = lean_unsigned_to_nat(1u);
v___x_425_ = l_Lean_Syntax_getArg(v_ref_422_, v___x_424_);
v___x_426_ = l_Lean_Syntax_getOptional_x3f(v___x_425_);
lean_dec(v___x_425_);
if (lean_obj_tag(v___x_426_) == 1)
{
uint8_t v_hasTrace_427_; 
lean_dec_ref(v_f_412_);
v_hasTrace_427_ = lean_ctor_get_uint8(v_options_421_, sizeof(void*)*1);
if (v_hasTrace_427_ == 0)
{
lean_dec_ref_known(v___x_426_, 1);
goto v___jp_418_;
}
else
{
lean_object* v_val_428_; lean_object* v___x_430_; uint8_t v_isShared_431_; uint8_t v_isSharedCheck_441_; 
v_val_428_ = lean_ctor_get(v___x_426_, 0);
v_isSharedCheck_441_ = !lean_is_exclusive(v___x_426_);
if (v_isSharedCheck_441_ == 0)
{
v___x_430_ = v___x_426_;
v_isShared_431_ = v_isSharedCheck_441_;
goto v_resetjp_429_;
}
else
{
lean_inc(v_val_428_);
lean_dec(v___x_426_);
v___x_430_ = lean_box(0);
v_isShared_431_ = v_isSharedCheck_441_;
goto v_resetjp_429_;
}
v_resetjp_429_:
{
lean_object* v___x_432_; lean_object* v___x_433_; uint8_t v___x_434_; 
v___x_432_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_AdaptationNote_0__initFn___closed__1_00___x40_Mathlib_Tactic_AdaptationNote_1330411147____hygCtx___hyg_2_));
v___x_433_ = lean_obj_once(&lp_mathlib_reportAdaptationNote___closed__1, &lp_mathlib_reportAdaptationNote___closed__1_once, _init_lp_mathlib_reportAdaptationNote___closed__1);
v___x_434_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_423_, v_options_421_, v___x_433_);
if (v___x_434_ == 0)
{
lean_del_object(v___x_430_);
lean_dec(v_val_428_);
goto v___jp_418_;
}
else
{
lean_object* v___x_435_; lean_object* v___x_437_; 
v___x_435_ = l_Lean_TSyntax_getDocString(v_val_428_);
lean_dec(v_val_428_);
if (v_isShared_431_ == 0)
{
lean_ctor_set_tag(v___x_430_, 3);
lean_ctor_set(v___x_430_, 0, v___x_435_);
v___x_437_ = v___x_430_;
goto v_reusejp_436_;
}
else
{
lean_object* v_reuseFailAlloc_440_; 
v_reuseFailAlloc_440_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v_reuseFailAlloc_440_, 0, v___x_435_);
v___x_437_ = v_reuseFailAlloc_440_;
goto v_reusejp_436_;
}
v_reusejp_436_:
{
lean_object* v___x_438_; lean_object* v___x_439_; 
v___x_438_ = l_Lean_MessageData_ofFormat(v___x_437_);
v___x_439_ = lp_mathlib_Lean_addTrace___at___00reportAdaptationNote_spec__0(v___x_432_, v___x_438_, v_a_413_, v_a_414_, v_a_415_, v_a_416_);
return v___x_439_;
}
}
}
}
}
else
{
lean_object* v___x_442_; lean_object* v___x_443_; 
lean_dec(v___x_426_);
v___x_442_ = lean_obj_once(&lp_mathlib_reportAdaptationNote___closed__4, &lp_mathlib_reportAdaptationNote___closed__4_once, _init_lp_mathlib_reportAdaptationNote___closed__4);
v___x_443_ = lp_mathlib_Lean_logError___at___00reportAdaptationNote_spec__1(v___x_442_, v_a_413_, v_a_414_, v_a_415_, v_a_416_);
if (lean_obj_tag(v___x_443_) == 0)
{
lean_object* v___x_445_; uint8_t v_isShared_446_; uint8_t v_isSharedCheck_474_; 
v_isSharedCheck_474_ = !lean_is_exclusive(v___x_443_);
if (v_isSharedCheck_474_ == 0)
{
lean_object* v_unused_475_; 
v_unused_475_ = lean_ctor_get(v___x_443_, 0);
lean_dec(v_unused_475_);
v___x_445_ = v___x_443_;
v_isShared_446_ = v_isSharedCheck_474_;
goto v_resetjp_444_;
}
else
{
lean_dec(v___x_443_);
v___x_445_ = lean_box(0);
v_isShared_446_ = v_isSharedCheck_474_;
goto v_resetjp_444_;
}
v_resetjp_444_:
{
lean_object* v___x_447_; lean_object* v___y_449_; lean_object* v___x_470_; lean_object* v___x_471_; 
v___x_447_ = lean_unsigned_to_nat(0u);
v___x_470_ = l_Lean_Syntax_getArg(v_ref_422_, v___x_447_);
v___x_471_ = l_Lean_Syntax_getTailInfo(v___x_470_);
lean_dec(v___x_470_);
if (lean_obj_tag(v___x_471_) == 0)
{
lean_object* v_trailing_472_; 
v_trailing_472_ = lean_ctor_get(v___x_471_, 2);
lean_inc_ref(v_trailing_472_);
lean_dec_ref_known(v___x_471_, 4);
v___y_449_ = v_trailing_472_;
goto v___jp_448_;
}
else
{
lean_object* v___x_473_; 
lean_dec(v___x_471_);
v___x_473_ = ((lean_object*)(lp_mathlib_reportAdaptationNote___closed__18));
v___y_449_ = v___x_473_;
goto v___jp_448_;
}
v___jp_448_:
{
lean_object* v___x_450_; lean_object* v___x_451_; lean_object* v___x_452_; lean_object* v___x_453_; lean_object* v___x_454_; lean_object* v___x_455_; lean_object* v___x_456_; lean_object* v___x_457_; lean_object* v___x_458_; lean_object* v___x_459_; lean_object* v___x_460_; lean_object* v___x_461_; lean_object* v___x_463_; 
v___x_450_ = lean_box(2);
v___x_451_ = lean_obj_once(&lp_mathlib_reportAdaptationNote___closed__14, &lp_mathlib_reportAdaptationNote___closed__14_once, _init_lp_mathlib_reportAdaptationNote___closed__14);
v___x_452_ = l_Lean_Syntax_updateTrailing(v___y_449_, v___x_451_);
v___x_453_ = l_Lean_Syntax_getArg(v_ref_422_, v___x_447_);
v___x_454_ = l_Lean_Syntax_unsetTrailing(v___x_453_);
lean_inc_n(v_ref_422_, 2);
v___x_455_ = l_Lean_Syntax_setArg(v_ref_422_, v___x_447_, v___x_454_);
v___x_456_ = lean_mk_empty_array_with_capacity(v___x_424_);
v___x_457_ = lean_array_push(v___x_456_, v___x_452_);
v___x_458_ = ((lean_object*)(lp_mathlib_reportAdaptationNote___closed__16));
v___x_459_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_459_, 0, v___x_450_);
lean_ctor_set(v___x_459_, 1, v___x_458_);
lean_ctor_set(v___x_459_, 2, v___x_457_);
v___x_460_ = l_Lean_Syntax_setArg(v___x_455_, v___x_424_, v___x_459_);
v___x_461_ = lean_apply_1(v_f_412_, v___x_460_);
if (v_isShared_446_ == 0)
{
lean_ctor_set_tag(v___x_445_, 1);
lean_ctor_set(v___x_445_, 0, v_ref_422_);
v___x_463_ = v___x_445_;
goto v_reusejp_462_;
}
else
{
lean_object* v_reuseFailAlloc_469_; 
v_reuseFailAlloc_469_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_469_, 0, v_ref_422_);
v___x_463_ = v_reuseFailAlloc_469_;
goto v_reusejp_462_;
}
v_reusejp_462_:
{
lean_object* v___x_464_; lean_object* v___x_465_; uint8_t v___x_466_; lean_object* v___x_467_; lean_object* v___x_468_; 
v___x_464_ = ((lean_object*)(lp_mathlib_reportAdaptationNote___closed__17));
v___x_465_ = lean_box(0);
v___x_466_ = 4;
v___x_467_ = l_Lean_MessageData_nil;
lean_inc(v_ref_422_);
v___x_468_ = l_Lean_Meta_Tactic_TryThis_addSuggestion(v_ref_422_, v___x_461_, v___x_463_, v___x_464_, v___x_465_, v___x_466_, v___x_467_, v_a_415_, v_a_416_);
return v___x_468_;
}
}
}
}
else
{
lean_dec_ref(v_f_412_);
return v___x_443_;
}
}
v___jp_418_:
{
lean_object* v___x_419_; lean_object* v___x_420_; 
v___x_419_ = lean_box(0);
v___x_420_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_420_, 0, v___x_419_);
return v___x_420_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_reportAdaptationNote___boxed(lean_object* v_f_476_, lean_object* v_a_477_, lean_object* v_a_478_, lean_object* v_a_479_, lean_object* v_a_480_, lean_object* v_a_481_){
_start:
{
lean_object* v_res_482_; 
v_res_482_ = lp_mathlib_reportAdaptationNote(v_f_476_, v_a_477_, v_a_478_, v_a_479_, v_a_480_);
lean_dec(v_a_480_);
lean_dec_ref(v_a_479_);
lean_dec(v_a_478_);
lean_dec_ref(v_a_477_);
return v_res_482_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__AdaptationNote______elabRules__adaptationNoteCmd__1_spec__0___redArg___closed__0(void){
_start:
{
lean_object* v___x_511_; lean_object* v___x_512_; lean_object* v___x_513_; 
v___x_511_ = lean_box(0);
v___x_512_ = l_Lean_Elab_unsupportedSyntaxExceptionId;
v___x_513_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_513_, 0, v___x_512_);
lean_ctor_set(v___x_513_, 1, v___x_511_);
return v___x_513_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__AdaptationNote______elabRules__adaptationNoteCmd__1_spec__0___redArg(){
_start:
{
lean_object* v___x_515_; lean_object* v___x_516_; 
v___x_515_ = lean_obj_once(&lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__AdaptationNote______elabRules__adaptationNoteCmd__1_spec__0___redArg___closed__0, &lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__AdaptationNote______elabRules__adaptationNoteCmd__1_spec__0___redArg___closed__0_once, _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__AdaptationNote______elabRules__adaptationNoteCmd__1_spec__0___redArg___closed__0);
v___x_516_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_516_, 0, v___x_515_);
return v___x_516_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__AdaptationNote______elabRules__adaptationNoteCmd__1_spec__0___redArg___boxed(lean_object* v___y_517_){
_start:
{
lean_object* v_res_518_; 
v_res_518_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__AdaptationNote______elabRules__adaptationNoteCmd__1_spec__0___redArg();
return v_res_518_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__AdaptationNote______elabRules__adaptationNoteCmd__1_spec__0(lean_object* v_00_u03b1_519_, lean_object* v___y_520_, lean_object* v___y_521_){
_start:
{
lean_object* v___x_523_; 
v___x_523_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__AdaptationNote______elabRules__adaptationNoteCmd__1_spec__0___redArg();
return v___x_523_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__AdaptationNote______elabRules__adaptationNoteCmd__1_spec__0___boxed(lean_object* v_00_u03b1_524_, lean_object* v___y_525_, lean_object* v___y_526_, lean_object* v___y_527_){
_start:
{
lean_object* v_res_528_; 
v_res_528_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__AdaptationNote______elabRules__adaptationNoteCmd__1_spec__0(v_00_u03b1_524_, v___y_525_, v___y_526_);
lean_dec(v___y_526_);
lean_dec_ref(v___y_525_);
return v_res_528_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Tactic__AdaptationNote______elabRules__adaptationNoteCmd__1___lam__0(lean_object* v_s_532_){
_start:
{
lean_object* v___x_533_; lean_object* v___x_534_; lean_object* v___x_535_; lean_object* v___x_536_; 
v___x_533_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Tactic__AdaptationNote______elabRules__adaptationNoteCmd__1___lam__0___closed__1));
v___x_534_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_534_, 0, v___x_533_);
lean_ctor_set(v___x_534_, 1, v_s_532_);
v___x_535_ = lean_box(0);
v___x_536_ = lean_alloc_ctor(0, 6, 0);
lean_ctor_set(v___x_536_, 0, v___x_534_);
lean_ctor_set(v___x_536_, 1, v___x_535_);
lean_ctor_set(v___x_536_, 2, v___x_535_);
lean_ctor_set(v___x_536_, 3, v___x_535_);
lean_ctor_set(v___x_536_, 4, v___x_535_);
lean_ctor_set(v___x_536_, 5, v___x_535_);
return v___x_536_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Tactic__AdaptationNote______elabRules__adaptationNoteCmd__1___lam__1(lean_object* v___f_537_, lean_object* v___y_538_, lean_object* v___y_539_, lean_object* v___y_540_, lean_object* v___y_541_, lean_object* v___y_542_, lean_object* v___y_543_){
_start:
{
lean_object* v___x_545_; 
v___x_545_ = lp_mathlib_reportAdaptationNote(v___f_537_, v___y_540_, v___y_541_, v___y_542_, v___y_543_);
return v___x_545_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Tactic__AdaptationNote______elabRules__adaptationNoteCmd__1___lam__1___boxed(lean_object* v___f_546_, lean_object* v___y_547_, lean_object* v___y_548_, lean_object* v___y_549_, lean_object* v___y_550_, lean_object* v___y_551_, lean_object* v___y_552_, lean_object* v___y_553_){
_start:
{
lean_object* v_res_554_; 
v_res_554_ = lp_mathlib___aux__Mathlib__Tactic__AdaptationNote______elabRules__adaptationNoteCmd__1___lam__1(v___f_546_, v___y_547_, v___y_548_, v___y_549_, v___y_550_, v___y_551_, v___y_552_);
lean_dec(v___y_552_);
lean_dec_ref(v___y_551_);
lean_dec(v___y_550_);
lean_dec_ref(v___y_549_);
lean_dec(v___y_548_);
lean_dec_ref(v___y_547_);
return v_res_554_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Tactic__AdaptationNote______elabRules__adaptationNoteCmd__1(lean_object* v_x_558_, lean_object* v_a_559_, lean_object* v_a_560_){
_start:
{
lean_object* v___x_562_; uint8_t v___x_563_; 
v___x_562_ = ((lean_object*)(lp_mathlib_adaptationNoteCmd___closed__1));
v___x_563_ = l_Lean_Syntax_isOfKind(v_x_558_, v___x_562_);
if (v___x_563_ == 0)
{
lean_object* v___x_564_; 
v___x_564_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__AdaptationNote______elabRules__adaptationNoteCmd__1_spec__0___redArg();
return v___x_564_;
}
else
{
lean_object* v___f_565_; lean_object* v___x_566_; 
v___f_565_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Tactic__AdaptationNote______elabRules__adaptationNoteCmd__1___closed__1));
v___x_566_ = l_Lean_Elab_Command_liftTermElabM___redArg(v___f_565_, v_a_559_, v_a_560_);
return v___x_566_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Tactic__AdaptationNote______elabRules__adaptationNoteCmd__1___boxed(lean_object* v_x_567_, lean_object* v_a_568_, lean_object* v_a_569_, lean_object* v_a_570_){
_start:
{
lean_object* v_res_571_; 
v_res_571_ = lp_mathlib___aux__Mathlib__Tactic__AdaptationNote______elabRules__adaptationNoteCmd__1(v_x_567_, v_a_568_, v_a_569_);
lean_dec(v_a_569_);
lean_dec_ref(v_a_568_);
return v_res_571_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__AdaptationNote______elabRules__tactic_x23adaptation__note____1_spec__0___redArg(){
_start:
{
lean_object* v___x_588_; lean_object* v___x_589_; 
v___x_588_ = lean_obj_once(&lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__AdaptationNote______elabRules__adaptationNoteCmd__1_spec__0___redArg___closed__0, &lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__AdaptationNote______elabRules__adaptationNoteCmd__1_spec__0___redArg___closed__0_once, _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__AdaptationNote______elabRules__adaptationNoteCmd__1_spec__0___redArg___closed__0);
v___x_589_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_589_, 0, v___x_588_);
return v___x_589_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__AdaptationNote______elabRules__tactic_x23adaptation__note____1_spec__0___redArg___boxed(lean_object* v___y_590_){
_start:
{
lean_object* v_res_591_; 
v_res_591_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__AdaptationNote______elabRules__tactic_x23adaptation__note____1_spec__0___redArg();
return v_res_591_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__AdaptationNote______elabRules__tactic_x23adaptation__note____1_spec__0(lean_object* v_00_u03b1_592_, lean_object* v___y_593_, lean_object* v___y_594_, lean_object* v___y_595_, lean_object* v___y_596_, lean_object* v___y_597_, lean_object* v___y_598_, lean_object* v___y_599_, lean_object* v___y_600_){
_start:
{
lean_object* v___x_602_; 
v___x_602_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__AdaptationNote______elabRules__tactic_x23adaptation__note____1_spec__0___redArg();
return v___x_602_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__AdaptationNote______elabRules__tactic_x23adaptation__note____1_spec__0___boxed(lean_object* v_00_u03b1_603_, lean_object* v___y_604_, lean_object* v___y_605_, lean_object* v___y_606_, lean_object* v___y_607_, lean_object* v___y_608_, lean_object* v___y_609_, lean_object* v___y_610_, lean_object* v___y_611_, lean_object* v___y_612_){
_start:
{
lean_object* v_res_613_; 
v_res_613_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__AdaptationNote______elabRules__tactic_x23adaptation__note____1_spec__0(v_00_u03b1_603_, v___y_604_, v___y_605_, v___y_606_, v___y_607_, v___y_608_, v___y_609_, v___y_610_, v___y_611_);
lean_dec(v___y_611_);
lean_dec_ref(v___y_610_);
lean_dec(v___y_609_);
lean_dec_ref(v___y_608_);
lean_dec(v___y_607_);
lean_dec_ref(v___y_606_);
lean_dec(v___y_605_);
lean_dec_ref(v___y_604_);
return v_res_613_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Tactic__AdaptationNote______elabRules__tactic_x23adaptation__note____1(lean_object* v_x_614_, lean_object* v_a_615_, lean_object* v_a_616_, lean_object* v_a_617_, lean_object* v_a_618_, lean_object* v_a_619_, lean_object* v_a_620_, lean_object* v_a_621_, lean_object* v_a_622_){
_start:
{
lean_object* v___x_624_; uint8_t v___x_625_; 
v___x_624_ = ((lean_object*)(lp_mathlib_tactic_x23adaptation__note___00__closed__1));
v___x_625_ = l_Lean_Syntax_isOfKind(v_x_614_, v___x_624_);
if (v___x_625_ == 0)
{
lean_object* v___x_626_; 
v___x_626_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__AdaptationNote______elabRules__tactic_x23adaptation__note____1_spec__0___redArg();
return v___x_626_;
}
else
{
lean_object* v___f_627_; lean_object* v___x_628_; 
v___f_627_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Tactic__AdaptationNote______elabRules__adaptationNoteCmd__1___closed__0));
v___x_628_ = lp_mathlib_reportAdaptationNote(v___f_627_, v_a_619_, v_a_620_, v_a_621_, v_a_622_);
return v___x_628_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Tactic__AdaptationNote______elabRules__tactic_x23adaptation__note____1___boxed(lean_object* v_x_629_, lean_object* v_a_630_, lean_object* v_a_631_, lean_object* v_a_632_, lean_object* v_a_633_, lean_object* v_a_634_, lean_object* v_a_635_, lean_object* v_a_636_, lean_object* v_a_637_, lean_object* v_a_638_){
_start:
{
lean_object* v_res_639_; 
v_res_639_ = lp_mathlib___aux__Mathlib__Tactic__AdaptationNote______elabRules__tactic_x23adaptation__note____1(v_x_629_, v_a_630_, v_a_631_, v_a_632_, v_a_633_, v_a_634_, v_a_635_, v_a_636_, v_a_637_);
lean_dec(v_a_637_);
lean_dec_ref(v_a_636_);
lean_dec(v_a_635_);
lean_dec_ref(v_a_634_);
lean_dec(v_a_633_);
lean_dec_ref(v_a_632_);
lean_dec(v_a_631_);
lean_dec_ref(v_a_630_);
return v_res_639_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00adaptationNoteTermElab_spec__0___redArg(){
_start:
{
lean_object* v___x_659_; lean_object* v___x_660_; 
v___x_659_ = lean_obj_once(&lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__AdaptationNote______elabRules__adaptationNoteCmd__1_spec__0___redArg___closed__0, &lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__AdaptationNote______elabRules__adaptationNoteCmd__1_spec__0___redArg___closed__0_once, _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__AdaptationNote______elabRules__adaptationNoteCmd__1_spec__0___redArg___closed__0);
v___x_660_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_660_, 0, v___x_659_);
return v___x_660_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00adaptationNoteTermElab_spec__0___redArg___boxed(lean_object* v___y_661_){
_start:
{
lean_object* v_res_662_; 
v_res_662_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00adaptationNoteTermElab_spec__0___redArg();
return v_res_662_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00adaptationNoteTermElab_spec__0(lean_object* v_00_u03b1_663_, lean_object* v___y_664_, lean_object* v___y_665_, lean_object* v___y_666_, lean_object* v___y_667_, lean_object* v___y_668_, lean_object* v___y_669_){
_start:
{
lean_object* v___x_671_; 
v___x_671_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00adaptationNoteTermElab_spec__0___redArg();
return v___x_671_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00adaptationNoteTermElab_spec__0___boxed(lean_object* v_00_u03b1_672_, lean_object* v___y_673_, lean_object* v___y_674_, lean_object* v___y_675_, lean_object* v___y_676_, lean_object* v___y_677_, lean_object* v___y_678_, lean_object* v___y_679_){
_start:
{
lean_object* v_res_680_; 
v_res_680_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00adaptationNoteTermElab_spec__0(v_00_u03b1_672_, v___y_673_, v___y_674_, v___y_675_, v___y_676_, v___y_677_, v___y_678_);
lean_dec(v___y_678_);
lean_dec_ref(v___y_677_);
lean_dec(v___y_676_);
lean_dec_ref(v___y_675_);
lean_dec(v___y_674_);
lean_dec_ref(v___y_673_);
return v_res_680_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_adaptationNoteTermElab___lam__0(lean_object* v___x_681_, lean_object* v_s_682_){
_start:
{
lean_object* v___x_683_; lean_object* v___x_684_; lean_object* v___x_685_; 
v___x_683_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_683_, 0, v___x_681_);
lean_ctor_set(v___x_683_, 1, v_s_682_);
v___x_684_ = lean_box(0);
v___x_685_ = lean_alloc_ctor(0, 6, 0);
lean_ctor_set(v___x_685_, 0, v___x_683_);
lean_ctor_set(v___x_685_, 1, v___x_684_);
lean_ctor_set(v___x_685_, 2, v___x_684_);
lean_ctor_set(v___x_685_, 3, v___x_684_);
lean_ctor_set(v___x_685_, 4, v___x_684_);
lean_ctor_set(v___x_685_, 5, v___x_684_);
return v___x_685_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_adaptationNoteTermElab(lean_object* v_x_688_, lean_object* v_a_689_, lean_object* v_a_690_, lean_object* v_a_691_, lean_object* v_a_692_, lean_object* v_a_693_, lean_object* v_a_694_, lean_object* v_a_695_){
_start:
{
lean_object* v___x_697_; uint8_t v___x_698_; lean_object* v___y_700_; lean_object* v___y_701_; lean_object* v___y_702_; lean_object* v___y_703_; lean_object* v___y_704_; lean_object* v___y_705_; lean_object* v___y_706_; 
v___x_697_ = ((lean_object*)(lp_mathlib_adaptationNoteTermStx___closed__1));
lean_inc(v_x_688_);
v___x_698_ = l_Lean_Syntax_isOfKind(v_x_688_, v___x_697_);
if (v___x_698_ == 0)
{
lean_object* v___x_720_; 
lean_dec(v_a_689_);
lean_dec(v_x_688_);
v___x_720_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00adaptationNoteTermElab_spec__0___redArg();
return v___x_720_;
}
else
{
lean_object* v___x_721_; lean_object* v___x_722_; uint8_t v___x_723_; 
v___x_721_ = lean_unsigned_to_nat(1u);
v___x_722_ = l_Lean_Syntax_getArg(v_x_688_, v___x_721_);
v___x_723_ = l_Lean_Syntax_isNone(v___x_722_);
if (v___x_723_ == 0)
{
uint8_t v___x_724_; 
v___x_724_ = l_Lean_Syntax_matchesNull(v___x_722_, v___x_721_);
if (v___x_724_ == 0)
{
lean_object* v___x_725_; 
lean_dec(v_a_689_);
lean_dec(v_x_688_);
v___x_725_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00adaptationNoteTermElab_spec__0___redArg();
return v___x_725_;
}
else
{
v___y_700_ = v_a_689_;
v___y_701_ = v_a_690_;
v___y_702_ = v_a_691_;
v___y_703_ = v_a_692_;
v___y_704_ = v_a_693_;
v___y_705_ = v_a_694_;
v___y_706_ = v_a_695_;
goto v___jp_699_;
}
}
else
{
lean_dec(v___x_722_);
v___y_700_ = v_a_689_;
v___y_701_ = v_a_690_;
v___y_702_ = v_a_691_;
v___y_703_ = v_a_692_;
v___y_704_ = v_a_693_;
v___y_705_ = v_a_694_;
v___y_706_ = v_a_695_;
goto v___jp_699_;
}
}
v___jp_699_:
{
lean_object* v___f_707_; lean_object* v___x_708_; 
v___f_707_ = ((lean_object*)(lp_mathlib_adaptationNoteTermElab___closed__0));
v___x_708_ = lp_mathlib_reportAdaptationNote(v___f_707_, v___y_703_, v___y_704_, v___y_705_, v___y_706_);
if (lean_obj_tag(v___x_708_) == 0)
{
lean_object* v___x_709_; lean_object* v___x_710_; lean_object* v___x_711_; 
lean_dec_ref_known(v___x_708_, 1);
v___x_709_ = lean_unsigned_to_nat(2u);
v___x_710_ = l_Lean_Syntax_getArg(v_x_688_, v___x_709_);
lean_dec(v_x_688_);
v___x_711_ = l_Lean_Elab_Term_elabTerm(v___x_710_, v___y_700_, v___x_698_, v___x_698_, v___y_701_, v___y_702_, v___y_703_, v___y_704_, v___y_705_, v___y_706_);
return v___x_711_;
}
else
{
lean_object* v_a_712_; lean_object* v___x_714_; uint8_t v_isShared_715_; uint8_t v_isSharedCheck_719_; 
lean_dec(v___y_700_);
lean_dec(v_x_688_);
v_a_712_ = lean_ctor_get(v___x_708_, 0);
v_isSharedCheck_719_ = !lean_is_exclusive(v___x_708_);
if (v_isSharedCheck_719_ == 0)
{
v___x_714_ = v___x_708_;
v_isShared_715_ = v_isSharedCheck_719_;
goto v_resetjp_713_;
}
else
{
lean_inc(v_a_712_);
lean_dec(v___x_708_);
v___x_714_ = lean_box(0);
v_isShared_715_ = v_isSharedCheck_719_;
goto v_resetjp_713_;
}
v_resetjp_713_:
{
lean_object* v___x_717_; 
if (v_isShared_715_ == 0)
{
v___x_717_ = v___x_714_;
goto v_reusejp_716_;
}
else
{
lean_object* v_reuseFailAlloc_718_; 
v_reuseFailAlloc_718_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_718_, 0, v_a_712_);
v___x_717_ = v_reuseFailAlloc_718_;
goto v_reusejp_716_;
}
v_reusejp_716_:
{
return v___x_717_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_adaptationNoteTermElab___boxed(lean_object* v_x_726_, lean_object* v_a_727_, lean_object* v_a_728_, lean_object* v_a_729_, lean_object* v_a_730_, lean_object* v_a_731_, lean_object* v_a_732_, lean_object* v_a_733_, lean_object* v_a_734_){
_start:
{
lean_object* v_res_735_; 
v_res_735_ = lp_mathlib_adaptationNoteTermElab(v_x_726_, v_a_727_, v_a_728_, v_a_729_, v_a_730_, v_a_731_, v_a_732_, v_a_733_);
lean_dec(v_a_733_);
lean_dec_ref(v_a_732_);
lean_dec(v_a_731_);
lean_dec_ref(v_a_730_);
lean_dec(v_a_729_);
lean_dec_ref(v_a_728_);
return v_res_735_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_Lean_Meta_TryThis(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Tactic_AdaptationNote(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Meta_TryThis(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_Lean_Meta_Tactic_TryThis(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Linter_Header(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Tactic_AdaptationNote(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Meta_Tactic_TryThis(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Linter_Header(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = lp_mathlib___private_Mathlib_Tactic_AdaptationNote_0__initFn_00___x40_Mathlib_Tactic_AdaptationNote_1330411147____hygCtx___hyg_2_();
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Lean_Meta_Tactic_TryThis(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Linter_Header(uint8_t builtin);
lean_object* initialize_Lean_Meta_TryThis(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_AdaptationNote(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Meta_Tactic_TryThis(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Linter_Header(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Meta_TryThis(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_AdaptationNote(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Tactic_AdaptationNote(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Tactic_AdaptationNote(builtin);
}
#ifdef __cplusplus
}
#endif
