// Lean compiler output
// Module: Mathlib.Control.Functor
// Imports: public import Init public meta import Init public import Mathlib.Data.Set.Defs import Mathlib.Tactic.Attr.Register
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
lean_object* l_Function_const___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_id___boxed(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_matchesNull(lean_object*, lean_object*);
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_String_toRawSubstring_x27(lean_object*);
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Functor_Const_mk___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Functor_Const_mk___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Functor_Const_mk(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Functor_Const_mk___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Functor_Const_mk_x27___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Functor_Const_mk_x27___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Functor_Const_mk_x27(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Functor_Const_mk_x27___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Functor_Const_run___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Functor_Const_run___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Functor_Const_run(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Functor_Const_run___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Functor_Const_map___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Functor_Const_map___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Functor_Const_map(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Functor_Const_map___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Functor_Const_functor___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Functor_Const_functor___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Functor_Const_functor___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Functor_Const_functor___lam__0___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Functor_Const_functor___closed__0 = (const lean_object*)&lp_mathlib_Functor_Const_functor___closed__0_value;
static const lean_closure_object lp_mathlib_Functor_Const_functor___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Functor_Const_map___boxed, .m_arity = 5, .m_num_fixed = 1, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib_Functor_Const_functor___closed__1 = (const lean_object*)&lp_mathlib_Functor_Const_functor___closed__1_value;
static const lean_ctor_object lp_mathlib_Functor_Const_functor___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Functor_Const_functor___closed__1_value),((lean_object*)&lp_mathlib_Functor_Const_functor___closed__0_value)}};
static const lean_object* lp_mathlib_Functor_Const_functor___closed__2 = (const lean_object*)&lp_mathlib_Functor_Const_functor___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_Functor_Const_functor(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Functor_Const_instInhabited___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Functor_Const_instInhabited___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Functor_Const_instInhabited(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Functor_Const_instInhabited___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Functor_AddConst_mk___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Functor_AddConst_mk___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Functor_AddConst_mk(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Functor_AddConst_mk___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Functor_AddConst_run___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Functor_AddConst_run___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Functor_AddConst_run(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Functor_AddConst_run___boxed(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Functor_AddConst_functor___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Functor_AddConst_functor___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Functor_AddConst_functor(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Functor_instInhabitedAddConst___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Functor_instInhabitedAddConst___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Functor_instInhabitedAddConst(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Functor_instInhabitedAddConst___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Functor_Comp_mk___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Functor_Comp_mk___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Functor_Comp_mk(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Functor_Comp_mk___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Functor_Comp_run___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Functor_Comp_run___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Functor_Comp_run(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Functor_Comp_run___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Functor_Comp_instInhabited___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Functor_Comp_instInhabited___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Functor_Comp_instInhabited(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Functor_Comp_instInhabited___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Functor_Comp_map___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Functor_Comp_map___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Functor_Comp_map(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Functor_Comp_functor___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Functor_Comp_functor___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Functor_Comp_functor(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Control_Functor_0__Functor_Comp_map_match__1_splitter___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Control_Functor_0__Functor_Comp_map_match__1_splitter(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Functor_Comp_seq___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Functor_Comp_seq___redArg___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Functor_Comp_seq___redArg___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Functor_Comp_seq___redArg___lam__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Functor_Comp_seq___redArg___lam__2___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Functor_Comp_seq___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Functor_Comp_seq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Functor_Comp_instPure___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Functor_Comp_instPure___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Functor_Comp_instPure(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Functor_Comp_instSeq___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Functor_Comp_instSeq___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Functor_Comp_instSeq(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Functor_Comp_instApplicativeComp___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Functor_Comp_instApplicativeComp___redArg___lam__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Function_const___boxed, .m_arity = 4, .m_num_fixed = 2, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib_Functor_Comp_instApplicativeComp___redArg___lam__1___closed__0 = (const lean_object*)&lp_mathlib_Functor_Comp_instApplicativeComp___redArg___lam__1___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Functor_Comp_instApplicativeComp___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Functor_Comp_instApplicativeComp___redArg___lam__2___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_id___boxed, .m_arity = 2, .m_num_fixed = 1, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib_Functor_Comp_instApplicativeComp___redArg___lam__2___closed__0 = (const lean_object*)&lp_mathlib_Functor_Comp_instApplicativeComp___redArg___lam__2___closed__0_value;
static const lean_closure_object lp_mathlib_Functor_Comp_instApplicativeComp___redArg___lam__2___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*3, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Function_const___boxed, .m_arity = 4, .m_num_fixed = 3, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Functor_Comp_instApplicativeComp___redArg___lam__2___closed__0_value)} };
static const lean_object* lp_mathlib_Functor_Comp_instApplicativeComp___redArg___lam__2___closed__1 = (const lean_object*)&lp_mathlib_Functor_Comp_instApplicativeComp___redArg___lam__2___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Functor_Comp_instApplicativeComp___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Functor_Comp_instApplicativeComp___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Functor_Comp_instApplicativeComp(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Functor_mapConstRev___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Functor_mapConstRev(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Functor_term___x24_x3e___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Functor"};
static const lean_object* lp_mathlib_Functor_term___x24_x3e___00__closed__0 = (const lean_object*)&lp_mathlib_Functor_term___x24_x3e___00__closed__0_value;
static const lean_string_object lp_mathlib_Functor_term___x24_x3e___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "term_$>_"};
static const lean_object* lp_mathlib_Functor_term___x24_x3e___00__closed__1 = (const lean_object*)&lp_mathlib_Functor_term___x24_x3e___00__closed__1_value;
static const lean_ctor_object lp_mathlib_Functor_term___x24_x3e___00__closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Functor_term___x24_x3e___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(39, 234, 35, 88, 204, 30, 230, 30)}};
static const lean_ctor_object lp_mathlib_Functor_term___x24_x3e___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Functor_term___x24_x3e___00__closed__2_value_aux_0),((lean_object*)&lp_mathlib_Functor_term___x24_x3e___00__closed__1_value),LEAN_SCALAR_PTR_LITERAL(255, 183, 87, 17, 192, 235, 188, 135)}};
static const lean_object* lp_mathlib_Functor_term___x24_x3e___00__closed__2 = (const lean_object*)&lp_mathlib_Functor_term___x24_x3e___00__closed__2_value;
static const lean_string_object lp_mathlib_Functor_term___x24_x3e___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_Functor_term___x24_x3e___00__closed__3 = (const lean_object*)&lp_mathlib_Functor_term___x24_x3e___00__closed__3_value;
static const lean_ctor_object lp_mathlib_Functor_term___x24_x3e___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Functor_term___x24_x3e___00__closed__3_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_Functor_term___x24_x3e___00__closed__4 = (const lean_object*)&lp_mathlib_Functor_term___x24_x3e___00__closed__4_value;
static const lean_string_object lp_mathlib_Functor_term___x24_x3e___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = " $> "};
static const lean_object* lp_mathlib_Functor_term___x24_x3e___00__closed__5 = (const lean_object*)&lp_mathlib_Functor_term___x24_x3e___00__closed__5_value;
static const lean_ctor_object lp_mathlib_Functor_term___x24_x3e___00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Functor_term___x24_x3e___00__closed__5_value)}};
static const lean_object* lp_mathlib_Functor_term___x24_x3e___00__closed__6 = (const lean_object*)&lp_mathlib_Functor_term___x24_x3e___00__closed__6_value;
static const lean_string_object lp_mathlib_Functor_term___x24_x3e___00__closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "term"};
static const lean_object* lp_mathlib_Functor_term___x24_x3e___00__closed__7 = (const lean_object*)&lp_mathlib_Functor_term___x24_x3e___00__closed__7_value;
static const lean_ctor_object lp_mathlib_Functor_term___x24_x3e___00__closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Functor_term___x24_x3e___00__closed__7_value),LEAN_SCALAR_PTR_LITERAL(187, 230, 181, 162, 253, 146, 122, 119)}};
static const lean_object* lp_mathlib_Functor_term___x24_x3e___00__closed__8 = (const lean_object*)&lp_mathlib_Functor_term___x24_x3e___00__closed__8_value;
static const lean_ctor_object lp_mathlib_Functor_term___x24_x3e___00__closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_Functor_term___x24_x3e___00__closed__8_value),((lean_object*)(((size_t)(101) << 1) | 1))}};
static const lean_object* lp_mathlib_Functor_term___x24_x3e___00__closed__9 = (const lean_object*)&lp_mathlib_Functor_term___x24_x3e___00__closed__9_value;
static const lean_ctor_object lp_mathlib_Functor_term___x24_x3e___00__closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Functor_term___x24_x3e___00__closed__4_value),((lean_object*)&lp_mathlib_Functor_term___x24_x3e___00__closed__6_value),((lean_object*)&lp_mathlib_Functor_term___x24_x3e___00__closed__9_value)}};
static const lean_object* lp_mathlib_Functor_term___x24_x3e___00__closed__10 = (const lean_object*)&lp_mathlib_Functor_term___x24_x3e___00__closed__10_value;
static const lean_ctor_object lp_mathlib_Functor_term___x24_x3e___00__closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 4}, .m_objs = {((lean_object*)&lp_mathlib_Functor_term___x24_x3e___00__closed__2_value),((lean_object*)(((size_t)(100) << 1) | 1)),((lean_object*)(((size_t)(101) << 1) | 1)),((lean_object*)&lp_mathlib_Functor_term___x24_x3e___00__closed__10_value)}};
static const lean_object* lp_mathlib_Functor_term___x24_x3e___00__closed__11 = (const lean_object*)&lp_mathlib_Functor_term___x24_x3e___00__closed__11_value;
LEAN_EXPORT const lean_object* lp_mathlib_Functor_term___x24_x3e__ = (const lean_object*)&lp_mathlib_Functor_term___x24_x3e___00__closed__11_value;
static const lean_string_object lp_mathlib_Functor___aux__Mathlib__Control__Functor______macroRules__Functor__term___x24_x3e____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib_Functor___aux__Mathlib__Control__Functor______macroRules__Functor__term___x24_x3e____1___closed__0 = (const lean_object*)&lp_mathlib_Functor___aux__Mathlib__Control__Functor______macroRules__Functor__term___x24_x3e____1___closed__0_value;
static const lean_string_object lp_mathlib_Functor___aux__Mathlib__Control__Functor______macroRules__Functor__term___x24_x3e____1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib_Functor___aux__Mathlib__Control__Functor______macroRules__Functor__term___x24_x3e____1___closed__1 = (const lean_object*)&lp_mathlib_Functor___aux__Mathlib__Control__Functor______macroRules__Functor__term___x24_x3e____1___closed__1_value;
static const lean_string_object lp_mathlib_Functor___aux__Mathlib__Control__Functor______macroRules__Functor__term___x24_x3e____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_mathlib_Functor___aux__Mathlib__Control__Functor______macroRules__Functor__term___x24_x3e____1___closed__2 = (const lean_object*)&lp_mathlib_Functor___aux__Mathlib__Control__Functor______macroRules__Functor__term___x24_x3e____1___closed__2_value;
static const lean_string_object lp_mathlib_Functor___aux__Mathlib__Control__Functor______macroRules__Functor__term___x24_x3e____1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "app"};
static const lean_object* lp_mathlib_Functor___aux__Mathlib__Control__Functor______macroRules__Functor__term___x24_x3e____1___closed__3 = (const lean_object*)&lp_mathlib_Functor___aux__Mathlib__Control__Functor______macroRules__Functor__term___x24_x3e____1___closed__3_value;
static const lean_ctor_object lp_mathlib_Functor___aux__Mathlib__Control__Functor______macroRules__Functor__term___x24_x3e____1___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Functor___aux__Mathlib__Control__Functor______macroRules__Functor__term___x24_x3e____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Functor___aux__Mathlib__Control__Functor______macroRules__Functor__term___x24_x3e____1___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Functor___aux__Mathlib__Control__Functor______macroRules__Functor__term___x24_x3e____1___closed__4_value_aux_0),((lean_object*)&lp_mathlib_Functor___aux__Mathlib__Control__Functor______macroRules__Functor__term___x24_x3e____1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Functor___aux__Mathlib__Control__Functor______macroRules__Functor__term___x24_x3e____1___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Functor___aux__Mathlib__Control__Functor______macroRules__Functor__term___x24_x3e____1___closed__4_value_aux_1),((lean_object*)&lp_mathlib_Functor___aux__Mathlib__Control__Functor______macroRules__Functor__term___x24_x3e____1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Functor___aux__Mathlib__Control__Functor______macroRules__Functor__term___x24_x3e____1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Functor___aux__Mathlib__Control__Functor______macroRules__Functor__term___x24_x3e____1___closed__4_value_aux_2),((lean_object*)&lp_mathlib_Functor___aux__Mathlib__Control__Functor______macroRules__Functor__term___x24_x3e____1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(69, 118, 10, 41, 220, 156, 243, 179)}};
static const lean_object* lp_mathlib_Functor___aux__Mathlib__Control__Functor______macroRules__Functor__term___x24_x3e____1___closed__4 = (const lean_object*)&lp_mathlib_Functor___aux__Mathlib__Control__Functor______macroRules__Functor__term___x24_x3e____1___closed__4_value;
static const lean_string_object lp_mathlib_Functor___aux__Mathlib__Control__Functor______macroRules__Functor__term___x24_x3e____1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "Functor.mapConstRev"};
static const lean_object* lp_mathlib_Functor___aux__Mathlib__Control__Functor______macroRules__Functor__term___x24_x3e____1___closed__5 = (const lean_object*)&lp_mathlib_Functor___aux__Mathlib__Control__Functor______macroRules__Functor__term___x24_x3e____1___closed__5_value;
static lean_once_cell_t lp_mathlib_Functor___aux__Mathlib__Control__Functor______macroRules__Functor__term___x24_x3e____1___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Functor___aux__Mathlib__Control__Functor______macroRules__Functor__term___x24_x3e____1___closed__6;
static const lean_string_object lp_mathlib_Functor___aux__Mathlib__Control__Functor______macroRules__Functor__term___x24_x3e____1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "mapConstRev"};
static const lean_object* lp_mathlib_Functor___aux__Mathlib__Control__Functor______macroRules__Functor__term___x24_x3e____1___closed__7 = (const lean_object*)&lp_mathlib_Functor___aux__Mathlib__Control__Functor______macroRules__Functor__term___x24_x3e____1___closed__7_value;
static const lean_ctor_object lp_mathlib_Functor___aux__Mathlib__Control__Functor______macroRules__Functor__term___x24_x3e____1___closed__8_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Functor_term___x24_x3e___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(39, 234, 35, 88, 204, 30, 230, 30)}};
static const lean_ctor_object lp_mathlib_Functor___aux__Mathlib__Control__Functor______macroRules__Functor__term___x24_x3e____1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Functor___aux__Mathlib__Control__Functor______macroRules__Functor__term___x24_x3e____1___closed__8_value_aux_0),((lean_object*)&lp_mathlib_Functor___aux__Mathlib__Control__Functor______macroRules__Functor__term___x24_x3e____1___closed__7_value),LEAN_SCALAR_PTR_LITERAL(96, 245, 103, 232, 185, 187, 148, 129)}};
static const lean_object* lp_mathlib_Functor___aux__Mathlib__Control__Functor______macroRules__Functor__term___x24_x3e____1___closed__8 = (const lean_object*)&lp_mathlib_Functor___aux__Mathlib__Control__Functor______macroRules__Functor__term___x24_x3e____1___closed__8_value;
static const lean_ctor_object lp_mathlib_Functor___aux__Mathlib__Control__Functor______macroRules__Functor__term___x24_x3e____1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Functor___aux__Mathlib__Control__Functor______macroRules__Functor__term___x24_x3e____1___closed__8_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Functor___aux__Mathlib__Control__Functor______macroRules__Functor__term___x24_x3e____1___closed__9 = (const lean_object*)&lp_mathlib_Functor___aux__Mathlib__Control__Functor______macroRules__Functor__term___x24_x3e____1___closed__9_value;
static const lean_ctor_object lp_mathlib_Functor___aux__Mathlib__Control__Functor______macroRules__Functor__term___x24_x3e____1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Functor___aux__Mathlib__Control__Functor______macroRules__Functor__term___x24_x3e____1___closed__9_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Functor___aux__Mathlib__Control__Functor______macroRules__Functor__term___x24_x3e____1___closed__10 = (const lean_object*)&lp_mathlib_Functor___aux__Mathlib__Control__Functor______macroRules__Functor__term___x24_x3e____1___closed__10_value;
static const lean_string_object lp_mathlib_Functor___aux__Mathlib__Control__Functor______macroRules__Functor__term___x24_x3e____1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib_Functor___aux__Mathlib__Control__Functor______macroRules__Functor__term___x24_x3e____1___closed__11 = (const lean_object*)&lp_mathlib_Functor___aux__Mathlib__Control__Functor______macroRules__Functor__term___x24_x3e____1___closed__11_value;
static const lean_ctor_object lp_mathlib_Functor___aux__Mathlib__Control__Functor______macroRules__Functor__term___x24_x3e____1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Functor___aux__Mathlib__Control__Functor______macroRules__Functor__term___x24_x3e____1___closed__11_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib_Functor___aux__Mathlib__Control__Functor______macroRules__Functor__term___x24_x3e____1___closed__12 = (const lean_object*)&lp_mathlib_Functor___aux__Mathlib__Control__Functor______macroRules__Functor__term___x24_x3e____1___closed__12_value;
LEAN_EXPORT lean_object* lp_mathlib_Functor___aux__Mathlib__Control__Functor______macroRules__Functor__term___x24_x3e____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Functor___aux__Mathlib__Control__Functor______macroRules__Functor__term___x24_x3e____1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Functor___aux__Mathlib__Control__Functor______unexpand__Functor__mapConstRev__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ident"};
static const lean_object* lp_mathlib_Functor___aux__Mathlib__Control__Functor______unexpand__Functor__mapConstRev__1___closed__0 = (const lean_object*)&lp_mathlib_Functor___aux__Mathlib__Control__Functor______unexpand__Functor__mapConstRev__1___closed__0_value;
static const lean_ctor_object lp_mathlib_Functor___aux__Mathlib__Control__Functor______unexpand__Functor__mapConstRev__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Functor___aux__Mathlib__Control__Functor______unexpand__Functor__mapConstRev__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(52, 159, 208, 51, 14, 60, 6, 71)}};
static const lean_object* lp_mathlib_Functor___aux__Mathlib__Control__Functor______unexpand__Functor__mapConstRev__1___closed__1 = (const lean_object*)&lp_mathlib_Functor___aux__Mathlib__Control__Functor______unexpand__Functor__mapConstRev__1___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Functor___aux__Mathlib__Control__Functor______unexpand__Functor__mapConstRev__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Functor___aux__Mathlib__Control__Functor______unexpand__Functor__mapConstRev__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Functor_Const_mk___redArg(lean_object* v_x_1_){
_start:
{
lean_inc(v_x_1_);
return v_x_1_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Functor_Const_mk___redArg___boxed(lean_object* v_x_2_){
_start:
{
lean_object* v_res_3_; 
v_res_3_ = lp_mathlib_Functor_Const_mk___redArg(v_x_2_);
lean_dec(v_x_2_);
return v_res_3_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Functor_Const_mk(lean_object* v_00_u03b1_4_, lean_object* v_00_u03b2_5_, lean_object* v_x_6_){
_start:
{
lean_inc(v_x_6_);
return v_x_6_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Functor_Const_mk___boxed(lean_object* v_00_u03b1_7_, lean_object* v_00_u03b2_8_, lean_object* v_x_9_){
_start:
{
lean_object* v_res_10_; 
v_res_10_ = lp_mathlib_Functor_Const_mk(v_00_u03b1_7_, v_00_u03b2_8_, v_x_9_);
lean_dec(v_x_9_);
return v_res_10_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Functor_Const_mk_x27___redArg(lean_object* v_x_11_){
_start:
{
lean_inc(v_x_11_);
return v_x_11_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Functor_Const_mk_x27___redArg___boxed(lean_object* v_x_12_){
_start:
{
lean_object* v_res_13_; 
v_res_13_ = lp_mathlib_Functor_Const_mk_x27___redArg(v_x_12_);
lean_dec(v_x_12_);
return v_res_13_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Functor_Const_mk_x27(lean_object* v_00_u03b1_14_, lean_object* v_x_15_){
_start:
{
lean_inc(v_x_15_);
return v_x_15_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Functor_Const_mk_x27___boxed(lean_object* v_00_u03b1_16_, lean_object* v_x_17_){
_start:
{
lean_object* v_res_18_; 
v_res_18_ = lp_mathlib_Functor_Const_mk_x27(v_00_u03b1_16_, v_x_17_);
lean_dec(v_x_17_);
return v_res_18_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Functor_Const_run___redArg(lean_object* v_x_19_){
_start:
{
lean_inc(v_x_19_);
return v_x_19_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Functor_Const_run___redArg___boxed(lean_object* v_x_20_){
_start:
{
lean_object* v_res_21_; 
v_res_21_ = lp_mathlib_Functor_Const_run___redArg(v_x_20_);
lean_dec(v_x_20_);
return v_res_21_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Functor_Const_run(lean_object* v_00_u03b1_22_, lean_object* v_00_u03b2_23_, lean_object* v_x_24_){
_start:
{
lean_inc(v_x_24_);
return v_x_24_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Functor_Const_run___boxed(lean_object* v_00_u03b1_25_, lean_object* v_00_u03b2_26_, lean_object* v_x_27_){
_start:
{
lean_object* v_res_28_; 
v_res_28_ = lp_mathlib_Functor_Const_run(v_00_u03b1_25_, v_00_u03b2_26_, v_x_27_);
lean_dec(v_x_27_);
return v_res_28_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Functor_Const_map___redArg(lean_object* v_x_29_){
_start:
{
lean_inc(v_x_29_);
return v_x_29_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Functor_Const_map___redArg___boxed(lean_object* v_x_30_){
_start:
{
lean_object* v_res_31_; 
v_res_31_ = lp_mathlib_Functor_Const_map___redArg(v_x_30_);
lean_dec(v_x_30_);
return v_res_31_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Functor_Const_map(lean_object* v_00_u03b3_32_, lean_object* v_00_u03b1_33_, lean_object* v_00_u03b2_34_, lean_object* v___f_35_, lean_object* v_x_36_){
_start:
{
lean_inc(v_x_36_);
return v_x_36_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Functor_Const_map___boxed(lean_object* v_00_u03b3_37_, lean_object* v_00_u03b1_38_, lean_object* v_00_u03b2_39_, lean_object* v___f_40_, lean_object* v_x_41_){
_start:
{
lean_object* v_res_42_; 
v_res_42_ = lp_mathlib_Functor_Const_map(v_00_u03b3_37_, v_00_u03b1_38_, v_00_u03b2_39_, v___f_40_, v_x_41_);
lean_dec(v_x_41_);
lean_dec(v___f_40_);
return v_res_42_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Functor_Const_functor___lam__0(lean_object* v_00_u03b1_43_, lean_object* v_00_u03b2_44_, lean_object* v___y_45_, lean_object* v___y_46_){
_start:
{
lean_inc(v___y_46_);
return v___y_46_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Functor_Const_functor___lam__0___boxed(lean_object* v_00_u03b1_47_, lean_object* v_00_u03b2_48_, lean_object* v___y_49_, lean_object* v___y_50_){
_start:
{
lean_object* v_res_51_; 
v_res_51_ = lp_mathlib_Functor_Const_functor___lam__0(v_00_u03b1_47_, v_00_u03b2_48_, v___y_49_, v___y_50_);
lean_dec(v___y_50_);
lean_dec(v___y_49_);
return v_res_51_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Functor_Const_functor(lean_object* v_00_u03b3_57_){
_start:
{
lean_object* v___x_58_; 
v___x_58_ = ((lean_object*)(lp_mathlib_Functor_Const_functor___closed__2));
return v___x_58_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Functor_Const_instInhabited___redArg(lean_object* v_inst_59_){
_start:
{
lean_inc(v_inst_59_);
return v_inst_59_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Functor_Const_instInhabited___redArg___boxed(lean_object* v_inst_60_){
_start:
{
lean_object* v_res_61_; 
v_res_61_ = lp_mathlib_Functor_Const_instInhabited___redArg(v_inst_60_);
lean_dec(v_inst_60_);
return v_res_61_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Functor_Const_instInhabited(lean_object* v_00_u03b1_62_, lean_object* v_00_u03b2_63_, lean_object* v_inst_64_){
_start:
{
lean_inc(v_inst_64_);
return v_inst_64_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Functor_Const_instInhabited___boxed(lean_object* v_00_u03b1_65_, lean_object* v_00_u03b2_66_, lean_object* v_inst_67_){
_start:
{
lean_object* v_res_68_; 
v_res_68_ = lp_mathlib_Functor_Const_instInhabited(v_00_u03b1_65_, v_00_u03b2_66_, v_inst_67_);
lean_dec(v_inst_67_);
return v_res_68_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Functor_AddConst_mk___redArg(lean_object* v_x_69_){
_start:
{
lean_inc(v_x_69_);
return v_x_69_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Functor_AddConst_mk___redArg___boxed(lean_object* v_x_70_){
_start:
{
lean_object* v_res_71_; 
v_res_71_ = lp_mathlib_Functor_AddConst_mk___redArg(v_x_70_);
lean_dec(v_x_70_);
return v_res_71_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Functor_AddConst_mk(lean_object* v_00_u03b1_72_, lean_object* v_00_u03b2_73_, lean_object* v_x_74_){
_start:
{
lean_inc(v_x_74_);
return v_x_74_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Functor_AddConst_mk___boxed(lean_object* v_00_u03b1_75_, lean_object* v_00_u03b2_76_, lean_object* v_x_77_){
_start:
{
lean_object* v_res_78_; 
v_res_78_ = lp_mathlib_Functor_AddConst_mk(v_00_u03b1_75_, v_00_u03b2_76_, v_x_77_);
lean_dec(v_x_77_);
return v_res_78_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Functor_AddConst_run___redArg(lean_object* v_a_79_){
_start:
{
lean_inc(v_a_79_);
return v_a_79_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Functor_AddConst_run___redArg___boxed(lean_object* v_a_80_){
_start:
{
lean_object* v_res_81_; 
v_res_81_ = lp_mathlib_Functor_AddConst_run___redArg(v_a_80_);
lean_dec(v_a_80_);
return v_res_81_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Functor_AddConst_run(lean_object* v_00_u03b1_82_, lean_object* v_00_u03b2_83_, lean_object* v_a_84_){
_start:
{
lean_inc(v_a_84_);
return v_a_84_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Functor_AddConst_run___boxed(lean_object* v_00_u03b1_85_, lean_object* v_00_u03b2_86_, lean_object* v_a_87_){
_start:
{
lean_object* v_res_88_; 
v_res_88_ = lp_mathlib_Functor_AddConst_run(v_00_u03b1_85_, v_00_u03b2_86_, v_a_87_);
lean_dec(v_a_87_);
return v_res_88_;
}
}
static lean_object* _init_lp_mathlib_Functor_AddConst_functor___closed__0(void){
_start:
{
lean_object* v___x_89_; 
v___x_89_ = lp_mathlib_Functor_Const_functor(lean_box(0));
return v___x_89_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Functor_AddConst_functor(lean_object* v___y_90_){
_start:
{
lean_object* v___x_91_; 
v___x_91_ = lean_obj_once(&lp_mathlib_Functor_AddConst_functor___closed__0, &lp_mathlib_Functor_AddConst_functor___closed__0_once, _init_lp_mathlib_Functor_AddConst_functor___closed__0);
return v___x_91_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Functor_instInhabitedAddConst___redArg(lean_object* v_inst_92_){
_start:
{
lean_inc(v_inst_92_);
return v_inst_92_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Functor_instInhabitedAddConst___redArg___boxed(lean_object* v_inst_93_){
_start:
{
lean_object* v_res_94_; 
v_res_94_ = lp_mathlib_Functor_instInhabitedAddConst___redArg(v_inst_93_);
lean_dec(v_inst_93_);
return v_res_94_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Functor_instInhabitedAddConst(lean_object* v_00_u03b1_95_, lean_object* v_00_u03b2_96_, lean_object* v_inst_97_){
_start:
{
lean_inc(v_inst_97_);
return v_inst_97_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Functor_instInhabitedAddConst___boxed(lean_object* v_00_u03b1_98_, lean_object* v_00_u03b2_99_, lean_object* v_inst_100_){
_start:
{
lean_object* v_res_101_; 
v_res_101_ = lp_mathlib_Functor_instInhabitedAddConst(v_00_u03b1_98_, v_00_u03b2_99_, v_inst_100_);
lean_dec(v_inst_100_);
return v_res_101_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Functor_Comp_mk___redArg(lean_object* v_x_102_){
_start:
{
lean_inc(v_x_102_);
return v_x_102_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Functor_Comp_mk___redArg___boxed(lean_object* v_x_103_){
_start:
{
lean_object* v_res_104_; 
v_res_104_ = lp_mathlib_Functor_Comp_mk___redArg(v_x_103_);
lean_dec(v_x_103_);
return v_res_104_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Functor_Comp_mk(lean_object* v_F_105_, lean_object* v_G_106_, lean_object* v_00_u03b1_107_, lean_object* v_x_108_){
_start:
{
lean_inc(v_x_108_);
return v_x_108_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Functor_Comp_mk___boxed(lean_object* v_F_109_, lean_object* v_G_110_, lean_object* v_00_u03b1_111_, lean_object* v_x_112_){
_start:
{
lean_object* v_res_113_; 
v_res_113_ = lp_mathlib_Functor_Comp_mk(v_F_109_, v_G_110_, v_00_u03b1_111_, v_x_112_);
lean_dec(v_x_112_);
return v_res_113_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Functor_Comp_run___redArg(lean_object* v_x_114_){
_start:
{
lean_inc(v_x_114_);
return v_x_114_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Functor_Comp_run___redArg___boxed(lean_object* v_x_115_){
_start:
{
lean_object* v_res_116_; 
v_res_116_ = lp_mathlib_Functor_Comp_run___redArg(v_x_115_);
lean_dec(v_x_115_);
return v_res_116_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Functor_Comp_run(lean_object* v_F_117_, lean_object* v_G_118_, lean_object* v_00_u03b1_119_, lean_object* v_x_120_){
_start:
{
lean_inc(v_x_120_);
return v_x_120_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Functor_Comp_run___boxed(lean_object* v_F_121_, lean_object* v_G_122_, lean_object* v_00_u03b1_123_, lean_object* v_x_124_){
_start:
{
lean_object* v_res_125_; 
v_res_125_ = lp_mathlib_Functor_Comp_run(v_F_121_, v_G_122_, v_00_u03b1_123_, v_x_124_);
lean_dec(v_x_124_);
return v_res_125_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Functor_Comp_instInhabited___redArg(lean_object* v_inst_126_){
_start:
{
lean_inc(v_inst_126_);
return v_inst_126_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Functor_Comp_instInhabited___redArg___boxed(lean_object* v_inst_127_){
_start:
{
lean_object* v_res_128_; 
v_res_128_ = lp_mathlib_Functor_Comp_instInhabited___redArg(v_inst_127_);
lean_dec(v_inst_127_);
return v_res_128_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Functor_Comp_instInhabited(lean_object* v_F_129_, lean_object* v_G_130_, lean_object* v_00_u03b1_131_, lean_object* v_inst_132_){
_start:
{
lean_inc(v_inst_132_);
return v_inst_132_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Functor_Comp_instInhabited___boxed(lean_object* v_F_133_, lean_object* v_G_134_, lean_object* v_00_u03b1_135_, lean_object* v_inst_136_){
_start:
{
lean_object* v_res_137_; 
v_res_137_ = lp_mathlib_Functor_Comp_instInhabited(v_F_133_, v_G_134_, v_00_u03b1_135_, v_inst_136_);
lean_dec(v_inst_136_);
return v_res_137_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Functor_Comp_map___redArg___lam__0(lean_object* v_inst_138_, lean_object* v_h_139_, lean_object* v_x_140_){
_start:
{
lean_object* v_map_141_; lean_object* v___x_142_; 
v_map_141_ = lean_ctor_get(v_inst_138_, 0);
lean_inc(v_map_141_);
lean_dec_ref(v_inst_138_);
v___x_142_ = lean_apply_4(v_map_141_, lean_box(0), lean_box(0), v_h_139_, v_x_140_);
return v___x_142_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Functor_Comp_map___redArg(lean_object* v_inst_143_, lean_object* v_inst_144_, lean_object* v_h_145_, lean_object* v_x_146_){
_start:
{
lean_object* v_map_147_; lean_object* v___f_148_; lean_object* v___x_149_; 
v_map_147_ = lean_ctor_get(v_inst_143_, 0);
lean_inc(v_map_147_);
lean_dec_ref(v_inst_143_);
v___f_148_ = lean_alloc_closure((void*)(lp_mathlib_Functor_Comp_map___redArg___lam__0), 3, 2);
lean_closure_set(v___f_148_, 0, v_inst_144_);
lean_closure_set(v___f_148_, 1, v_h_145_);
v___x_149_ = lean_apply_4(v_map_147_, lean_box(0), lean_box(0), v___f_148_, v_x_146_);
return v___x_149_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Functor_Comp_map(lean_object* v_F_150_, lean_object* v_G_151_, lean_object* v_inst_152_, lean_object* v_inst_153_, lean_object* v_00_u03b1_154_, lean_object* v_00_u03b2_155_, lean_object* v_h_156_, lean_object* v_x_157_){
_start:
{
lean_object* v___x_158_; 
v___x_158_ = lp_mathlib_Functor_Comp_map___redArg(v_inst_152_, v_inst_153_, v_h_156_, v_x_157_);
return v___x_158_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Functor_Comp_functor___redArg___lam__0(lean_object* v_inst_159_, lean_object* v_inst_160_, lean_object* v_00_u03b1_161_, lean_object* v_00_u03b2_162_, lean_object* v___y_163_, lean_object* v___y_164_){
_start:
{
lean_object* v___x_165_; lean_object* v___x_166_; 
v___x_165_ = lean_alloc_closure((void*)(l_Function_const___boxed), 4, 3);
lean_closure_set(v___x_165_, 0, lean_box(0));
lean_closure_set(v___x_165_, 1, lean_box(0));
lean_closure_set(v___x_165_, 2, v___y_163_);
v___x_166_ = lp_mathlib_Functor_Comp_map___redArg(v_inst_159_, v_inst_160_, v___x_165_, v___y_164_);
return v___x_166_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Functor_Comp_functor___redArg(lean_object* v_inst_167_, lean_object* v_inst_168_){
_start:
{
lean_object* v___f_169_; lean_object* v___x_170_; lean_object* v___x_171_; 
lean_inc_ref(v_inst_168_);
lean_inc_ref(v_inst_167_);
v___f_169_ = lean_alloc_closure((void*)(lp_mathlib_Functor_Comp_functor___redArg___lam__0), 6, 2);
lean_closure_set(v___f_169_, 0, v_inst_167_);
lean_closure_set(v___f_169_, 1, v_inst_168_);
v___x_170_ = lean_alloc_closure((void*)(lp_mathlib_Functor_Comp_map), 8, 4);
lean_closure_set(v___x_170_, 0, lean_box(0));
lean_closure_set(v___x_170_, 1, lean_box(0));
lean_closure_set(v___x_170_, 2, v_inst_167_);
lean_closure_set(v___x_170_, 3, v_inst_168_);
v___x_171_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_171_, 0, v___x_170_);
lean_ctor_set(v___x_171_, 1, v___f_169_);
return v___x_171_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Functor_Comp_functor(lean_object* v_F_172_, lean_object* v_G_173_, lean_object* v_inst_174_, lean_object* v_inst_175_){
_start:
{
lean_object* v___x_176_; 
v___x_176_ = lp_mathlib_Functor_Comp_functor___redArg(v_inst_174_, v_inst_175_);
return v___x_176_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Control_Functor_0__Functor_Comp_map_match__1_splitter___redArg(lean_object* v_x_177_, lean_object* v_h__1_178_){
_start:
{
lean_object* v___x_179_; 
v___x_179_ = lean_apply_1(v_h__1_178_, v_x_177_);
return v___x_179_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Control_Functor_0__Functor_Comp_map_match__1_splitter(lean_object* v_F_180_, lean_object* v_G_181_, lean_object* v_00_u03b1_182_, lean_object* v_motive_183_, lean_object* v_x_184_, lean_object* v_h__1_185_){
_start:
{
lean_object* v___x_186_; 
v___x_186_ = lean_apply_1(v_h__1_185_, v_x_184_);
return v___x_186_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Functor_Comp_seq___redArg___lam__0(lean_object* v_x2_187_, lean_object* v_x_188_){
_start:
{
lean_inc(v_x2_187_);
return v_x2_187_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Functor_Comp_seq___redArg___lam__0___boxed(lean_object* v_x2_189_, lean_object* v_x_190_){
_start:
{
lean_object* v_res_191_; 
v_res_191_ = lp_mathlib_Functor_Comp_seq___redArg___lam__0(v_x2_189_, v_x_190_);
lean_dec(v_x2_189_);
return v_res_191_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Functor_Comp_seq___redArg___lam__1(lean_object* v_toSeq_192_, lean_object* v_x1_193_, lean_object* v_x2_194_){
_start:
{
lean_object* v___f_195_; lean_object* v___x_196_; 
v___f_195_ = lean_alloc_closure((void*)(lp_mathlib_Functor_Comp_seq___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_195_, 0, v_x2_194_);
v___x_196_ = lean_apply_4(v_toSeq_192_, lean_box(0), lean_box(0), v_x1_193_, v___f_195_);
return v___x_196_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Functor_Comp_seq___redArg___lam__2(lean_object* v___x_197_, lean_object* v_x_198_){
_start:
{
lean_inc(v___x_197_);
return v___x_197_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Functor_Comp_seq___redArg___lam__2___boxed(lean_object* v___x_199_, lean_object* v_x_200_){
_start:
{
lean_object* v_res_201_; 
v_res_201_ = lp_mathlib_Functor_Comp_seq___redArg___lam__2(v___x_199_, v_x_200_);
lean_dec(v___x_199_);
return v_res_201_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Functor_Comp_seq___redArg(lean_object* v_inst_202_, lean_object* v_inst_203_, lean_object* v_x_204_, lean_object* v_x_205_){
_start:
{
lean_object* v_toFunctor_206_; lean_object* v_toSeq_207_; lean_object* v_toSeq_208_; lean_object* v_map_209_; lean_object* v___f_210_; lean_object* v___x_211_; lean_object* v___x_212_; lean_object* v___f_213_; lean_object* v___x_214_; lean_object* v___x_215_; 
v_toFunctor_206_ = lean_ctor_get(v_inst_202_, 0);
lean_inc_ref(v_toFunctor_206_);
v_toSeq_207_ = lean_ctor_get(v_inst_202_, 2);
lean_inc(v_toSeq_207_);
lean_dec_ref(v_inst_202_);
v_toSeq_208_ = lean_ctor_get(v_inst_203_, 2);
lean_inc(v_toSeq_208_);
lean_dec_ref(v_inst_203_);
v_map_209_ = lean_ctor_get(v_toFunctor_206_, 0);
lean_inc(v_map_209_);
lean_dec_ref(v_toFunctor_206_);
v___f_210_ = lean_alloc_closure((void*)(lp_mathlib_Functor_Comp_seq___redArg___lam__1), 3, 1);
lean_closure_set(v___f_210_, 0, v_toSeq_208_);
v___x_211_ = lean_box(0);
v___x_212_ = lean_apply_1(v_x_205_, v___x_211_);
v___f_213_ = lean_alloc_closure((void*)(lp_mathlib_Functor_Comp_seq___redArg___lam__2___boxed), 2, 1);
lean_closure_set(v___f_213_, 0, v___x_212_);
v___x_214_ = lean_apply_4(v_map_209_, lean_box(0), lean_box(0), v___f_210_, v_x_204_);
v___x_215_ = lean_apply_4(v_toSeq_207_, lean_box(0), lean_box(0), v___x_214_, v___f_213_);
return v___x_215_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Functor_Comp_seq(lean_object* v_F_216_, lean_object* v_G_217_, lean_object* v_inst_218_, lean_object* v_inst_219_, lean_object* v_00_u03b1_220_, lean_object* v_00_u03b2_221_, lean_object* v_x_222_, lean_object* v_x_223_){
_start:
{
lean_object* v___x_224_; 
v___x_224_ = lp_mathlib_Functor_Comp_seq___redArg(v_inst_218_, v_inst_219_, v_x_222_, v_x_223_);
return v___x_224_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Functor_Comp_instPure___redArg___lam__0(lean_object* v_toPure_225_, lean_object* v_toPure_226_, lean_object* v_00_u03b1_227_, lean_object* v_x_228_){
_start:
{
lean_object* v___x_229_; lean_object* v___x_230_; 
v___x_229_ = lean_apply_2(v_toPure_225_, lean_box(0), v_x_228_);
v___x_230_ = lean_apply_2(v_toPure_226_, lean_box(0), v___x_229_);
return v___x_230_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Functor_Comp_instPure___redArg(lean_object* v_inst_231_, lean_object* v_inst_232_){
_start:
{
lean_object* v_toPure_233_; lean_object* v_toPure_234_; lean_object* v___f_235_; 
v_toPure_233_ = lean_ctor_get(v_inst_231_, 1);
lean_inc(v_toPure_233_);
lean_dec_ref(v_inst_231_);
v_toPure_234_ = lean_ctor_get(v_inst_232_, 1);
lean_inc(v_toPure_234_);
lean_dec_ref(v_inst_232_);
v___f_235_ = lean_alloc_closure((void*)(lp_mathlib_Functor_Comp_instPure___redArg___lam__0), 4, 2);
lean_closure_set(v___f_235_, 0, v_toPure_234_);
lean_closure_set(v___f_235_, 1, v_toPure_233_);
return v___f_235_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Functor_Comp_instPure(lean_object* v_F_236_, lean_object* v_G_237_, lean_object* v_inst_238_, lean_object* v_inst_239_){
_start:
{
lean_object* v___x_240_; 
v___x_240_ = lp_mathlib_Functor_Comp_instPure___redArg(v_inst_238_, v_inst_239_);
return v___x_240_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Functor_Comp_instSeq___redArg___lam__0(lean_object* v_inst_241_, lean_object* v_inst_242_, lean_object* v_00_u03b1_243_, lean_object* v_00_u03b2_244_, lean_object* v_f_245_, lean_object* v_x_246_){
_start:
{
lean_object* v___x_247_; 
v___x_247_ = lp_mathlib_Functor_Comp_seq___redArg(v_inst_241_, v_inst_242_, v_f_245_, v_x_246_);
return v___x_247_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Functor_Comp_instSeq___redArg(lean_object* v_inst_248_, lean_object* v_inst_249_){
_start:
{
lean_object* v___f_250_; 
v___f_250_ = lean_alloc_closure((void*)(lp_mathlib_Functor_Comp_instSeq___redArg___lam__0), 6, 2);
lean_closure_set(v___f_250_, 0, v_inst_248_);
lean_closure_set(v___f_250_, 1, v_inst_249_);
return v___f_250_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Functor_Comp_instSeq(lean_object* v_F_251_, lean_object* v_G_252_, lean_object* v_inst_253_, lean_object* v_inst_254_){
_start:
{
lean_object* v___f_255_; 
v___f_255_ = lean_alloc_closure((void*)(lp_mathlib_Functor_Comp_instSeq___redArg___lam__0), 6, 2);
lean_closure_set(v___f_255_, 0, v_inst_253_);
lean_closure_set(v___f_255_, 1, v_inst_254_);
return v___f_255_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Functor_Comp_instApplicativeComp___redArg___lam__0(lean_object* v_toFunctor_256_, lean_object* v_toFunctor_257_, lean_object* v_00_u03b1_258_, lean_object* v_00_u03b2_259_, lean_object* v___y_260_, lean_object* v___y_261_){
_start:
{
lean_object* v___x_262_; lean_object* v___x_263_; 
v___x_262_ = lean_alloc_closure((void*)(l_Function_const___boxed), 4, 3);
lean_closure_set(v___x_262_, 0, lean_box(0));
lean_closure_set(v___x_262_, 1, lean_box(0));
lean_closure_set(v___x_262_, 2, v___y_260_);
v___x_263_ = lp_mathlib_Functor_Comp_map___redArg(v_toFunctor_256_, v_toFunctor_257_, v___x_262_, v___y_261_);
return v___x_263_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Functor_Comp_instApplicativeComp___redArg___lam__1(lean_object* v_toFunctor_265_, lean_object* v_toFunctor_266_, lean_object* v_inst_267_, lean_object* v_inst_268_, lean_object* v_00_u03b1_269_, lean_object* v_00_u03b2_270_, lean_object* v_a_271_, lean_object* v_b_272_){
_start:
{
lean_object* v___x_273_; lean_object* v___x_274_; lean_object* v___x_275_; 
v___x_273_ = ((lean_object*)(lp_mathlib_Functor_Comp_instApplicativeComp___redArg___lam__1___closed__0));
v___x_274_ = lp_mathlib_Functor_Comp_map___redArg(v_toFunctor_265_, v_toFunctor_266_, v___x_273_, v_a_271_);
v___x_275_ = lp_mathlib_Functor_Comp_seq___redArg(v_inst_267_, v_inst_268_, v___x_274_, v_b_272_);
return v___x_275_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Functor_Comp_instApplicativeComp___redArg___lam__2(lean_object* v_toFunctor_279_, lean_object* v_toFunctor_280_, lean_object* v_inst_281_, lean_object* v_inst_282_, lean_object* v_00_u03b1_283_, lean_object* v_00_u03b2_284_, lean_object* v_a_285_, lean_object* v_b_286_){
_start:
{
lean_object* v___x_287_; lean_object* v___x_288_; lean_object* v___x_289_; 
v___x_287_ = ((lean_object*)(lp_mathlib_Functor_Comp_instApplicativeComp___redArg___lam__2___closed__1));
v___x_288_ = lp_mathlib_Functor_Comp_map___redArg(v_toFunctor_279_, v_toFunctor_280_, v___x_287_, v_a_285_);
v___x_289_ = lp_mathlib_Functor_Comp_seq___redArg(v_inst_281_, v_inst_282_, v___x_288_, v_b_286_);
return v___x_289_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Functor_Comp_instApplicativeComp___redArg(lean_object* v_inst_290_, lean_object* v_inst_291_){
_start:
{
lean_object* v_toFunctor_292_; lean_object* v_toFunctor_293_; lean_object* v___f_294_; lean_object* v___f_295_; lean_object* v___f_296_; lean_object* v___x_297_; lean_object* v___x_298_; lean_object* v___x_299_; lean_object* v___x_300_; lean_object* v___x_301_; 
v_toFunctor_292_ = lean_ctor_get(v_inst_290_, 0);
v_toFunctor_293_ = lean_ctor_get(v_inst_291_, 0);
lean_inc_ref_n(v_toFunctor_293_, 4);
lean_inc_ref_n(v_toFunctor_292_, 4);
v___f_294_ = lean_alloc_closure((void*)(lp_mathlib_Functor_Comp_instApplicativeComp___redArg___lam__0), 6, 2);
lean_closure_set(v___f_294_, 0, v_toFunctor_292_);
lean_closure_set(v___f_294_, 1, v_toFunctor_293_);
lean_inc_ref_n(v_inst_291_, 3);
lean_inc_ref_n(v_inst_290_, 3);
v___f_295_ = lean_alloc_closure((void*)(lp_mathlib_Functor_Comp_instApplicativeComp___redArg___lam__1), 8, 4);
lean_closure_set(v___f_295_, 0, v_toFunctor_292_);
lean_closure_set(v___f_295_, 1, v_toFunctor_293_);
lean_closure_set(v___f_295_, 2, v_inst_290_);
lean_closure_set(v___f_295_, 3, v_inst_291_);
v___f_296_ = lean_alloc_closure((void*)(lp_mathlib_Functor_Comp_instApplicativeComp___redArg___lam__2), 8, 4);
lean_closure_set(v___f_296_, 0, v_toFunctor_292_);
lean_closure_set(v___f_296_, 1, v_toFunctor_293_);
lean_closure_set(v___f_296_, 2, v_inst_290_);
lean_closure_set(v___f_296_, 3, v_inst_291_);
v___x_297_ = lean_alloc_closure((void*)(lp_mathlib_Functor_Comp_map), 8, 4);
lean_closure_set(v___x_297_, 0, lean_box(0));
lean_closure_set(v___x_297_, 1, lean_box(0));
lean_closure_set(v___x_297_, 2, v_toFunctor_292_);
lean_closure_set(v___x_297_, 3, v_toFunctor_293_);
v___x_298_ = lp_mathlib_Functor_Comp_instPure___redArg(v_inst_290_, v_inst_291_);
v___x_299_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_299_, 0, v___x_297_);
lean_ctor_set(v___x_299_, 1, v___f_294_);
v___x_300_ = lean_alloc_closure((void*)(lp_mathlib_Functor_Comp_seq), 8, 4);
lean_closure_set(v___x_300_, 0, lean_box(0));
lean_closure_set(v___x_300_, 1, lean_box(0));
lean_closure_set(v___x_300_, 2, v_inst_290_);
lean_closure_set(v___x_300_, 3, v_inst_291_);
v___x_301_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_301_, 0, v___x_299_);
lean_ctor_set(v___x_301_, 1, v___x_298_);
lean_ctor_set(v___x_301_, 2, v___x_300_);
lean_ctor_set(v___x_301_, 3, v___f_295_);
lean_ctor_set(v___x_301_, 4, v___f_296_);
return v___x_301_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Functor_Comp_instApplicativeComp(lean_object* v_F_302_, lean_object* v_G_303_, lean_object* v_inst_304_, lean_object* v_inst_305_){
_start:
{
lean_object* v___x_306_; 
v___x_306_ = lp_mathlib_Functor_Comp_instApplicativeComp___redArg(v_inst_304_, v_inst_305_);
return v___x_306_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Functor_mapConstRev___redArg(lean_object* v_inst_307_, lean_object* v_a_308_, lean_object* v_b_309_){
_start:
{
lean_object* v_mapConst_310_; lean_object* v___x_311_; 
v_mapConst_310_ = lean_ctor_get(v_inst_307_, 1);
lean_inc(v_mapConst_310_);
lean_dec_ref(v_inst_307_);
v___x_311_ = lean_apply_4(v_mapConst_310_, lean_box(0), lean_box(0), v_b_309_, v_a_308_);
return v___x_311_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Functor_mapConstRev(lean_object* v_f_312_, lean_object* v_inst_313_, lean_object* v_00_u03b1_314_, lean_object* v_00_u03b2_315_, lean_object* v_a_316_, lean_object* v_b_317_){
_start:
{
lean_object* v_mapConst_318_; lean_object* v___x_319_; 
v_mapConst_318_ = lean_ctor_get(v_inst_313_, 1);
lean_inc(v_mapConst_318_);
lean_dec_ref(v_inst_313_);
v___x_319_ = lean_apply_4(v_mapConst_318_, lean_box(0), lean_box(0), v_b_317_, v_a_316_);
return v___x_319_;
}
}
static lean_object* _init_lp_mathlib_Functor___aux__Mathlib__Control__Functor______macroRules__Functor__term___x24_x3e____1___closed__6(void){
_start:
{
lean_object* v___x_357_; lean_object* v___x_358_; 
v___x_357_ = ((lean_object*)(lp_mathlib_Functor___aux__Mathlib__Control__Functor______macroRules__Functor__term___x24_x3e____1___closed__5));
v___x_358_ = l_String_toRawSubstring_x27(v___x_357_);
return v___x_358_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Functor___aux__Mathlib__Control__Functor______macroRules__Functor__term___x24_x3e____1(lean_object* v_x_372_, lean_object* v_a_373_, lean_object* v_a_374_){
_start:
{
lean_object* v___x_375_; uint8_t v___x_376_; 
v___x_375_ = ((lean_object*)(lp_mathlib_Functor_term___x24_x3e___00__closed__2));
lean_inc(v_x_372_);
v___x_376_ = l_Lean_Syntax_isOfKind(v_x_372_, v___x_375_);
if (v___x_376_ == 0)
{
lean_object* v___x_377_; lean_object* v___x_378_; 
lean_dec(v_x_372_);
v___x_377_ = lean_box(1);
v___x_378_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_378_, 0, v___x_377_);
lean_ctor_set(v___x_378_, 1, v_a_374_);
return v___x_378_;
}
else
{
lean_object* v_quotContext_379_; lean_object* v_currMacroScope_380_; lean_object* v_ref_381_; lean_object* v___x_382_; lean_object* v___x_383_; lean_object* v___x_384_; lean_object* v___x_385_; uint8_t v___x_386_; lean_object* v___x_387_; lean_object* v___x_388_; lean_object* v___x_389_; lean_object* v___x_390_; lean_object* v___x_391_; lean_object* v___x_392_; lean_object* v___x_393_; lean_object* v___x_394_; lean_object* v___x_395_; lean_object* v___x_396_; lean_object* v___x_397_; 
v_quotContext_379_ = lean_ctor_get(v_a_373_, 1);
v_currMacroScope_380_ = lean_ctor_get(v_a_373_, 2);
v_ref_381_ = lean_ctor_get(v_a_373_, 5);
v___x_382_ = lean_unsigned_to_nat(0u);
v___x_383_ = l_Lean_Syntax_getArg(v_x_372_, v___x_382_);
v___x_384_ = lean_unsigned_to_nat(2u);
v___x_385_ = l_Lean_Syntax_getArg(v_x_372_, v___x_384_);
lean_dec(v_x_372_);
v___x_386_ = 0;
v___x_387_ = l_Lean_SourceInfo_fromRef(v_ref_381_, v___x_386_);
v___x_388_ = ((lean_object*)(lp_mathlib_Functor___aux__Mathlib__Control__Functor______macroRules__Functor__term___x24_x3e____1___closed__4));
v___x_389_ = lean_obj_once(&lp_mathlib_Functor___aux__Mathlib__Control__Functor______macroRules__Functor__term___x24_x3e____1___closed__6, &lp_mathlib_Functor___aux__Mathlib__Control__Functor______macroRules__Functor__term___x24_x3e____1___closed__6_once, _init_lp_mathlib_Functor___aux__Mathlib__Control__Functor______macroRules__Functor__term___x24_x3e____1___closed__6);
v___x_390_ = ((lean_object*)(lp_mathlib_Functor___aux__Mathlib__Control__Functor______macroRules__Functor__term___x24_x3e____1___closed__8));
lean_inc(v_currMacroScope_380_);
lean_inc(v_quotContext_379_);
v___x_391_ = l_Lean_addMacroScope(v_quotContext_379_, v___x_390_, v_currMacroScope_380_);
v___x_392_ = ((lean_object*)(lp_mathlib_Functor___aux__Mathlib__Control__Functor______macroRules__Functor__term___x24_x3e____1___closed__10));
lean_inc_n(v___x_387_, 2);
v___x_393_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_393_, 0, v___x_387_);
lean_ctor_set(v___x_393_, 1, v___x_389_);
lean_ctor_set(v___x_393_, 2, v___x_391_);
lean_ctor_set(v___x_393_, 3, v___x_392_);
v___x_394_ = ((lean_object*)(lp_mathlib_Functor___aux__Mathlib__Control__Functor______macroRules__Functor__term___x24_x3e____1___closed__12));
v___x_395_ = l_Lean_Syntax_node2(v___x_387_, v___x_394_, v___x_383_, v___x_385_);
v___x_396_ = l_Lean_Syntax_node2(v___x_387_, v___x_388_, v___x_393_, v___x_395_);
v___x_397_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_397_, 0, v___x_396_);
lean_ctor_set(v___x_397_, 1, v_a_374_);
return v___x_397_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Functor___aux__Mathlib__Control__Functor______macroRules__Functor__term___x24_x3e____1___boxed(lean_object* v_x_398_, lean_object* v_a_399_, lean_object* v_a_400_){
_start:
{
lean_object* v_res_401_; 
v_res_401_ = lp_mathlib_Functor___aux__Mathlib__Control__Functor______macroRules__Functor__term___x24_x3e____1(v_x_398_, v_a_399_, v_a_400_);
lean_dec_ref(v_a_399_);
return v_res_401_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Functor___aux__Mathlib__Control__Functor______unexpand__Functor__mapConstRev__1(lean_object* v_x_405_, lean_object* v_a_406_, lean_object* v_a_407_){
_start:
{
lean_object* v___x_408_; uint8_t v___x_409_; 
v___x_408_ = ((lean_object*)(lp_mathlib_Functor___aux__Mathlib__Control__Functor______macroRules__Functor__term___x24_x3e____1___closed__4));
lean_inc(v_x_405_);
v___x_409_ = l_Lean_Syntax_isOfKind(v_x_405_, v___x_408_);
if (v___x_409_ == 0)
{
lean_object* v___x_410_; lean_object* v___x_411_; 
lean_dec(v_x_405_);
v___x_410_ = lean_box(0);
v___x_411_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_411_, 0, v___x_410_);
lean_ctor_set(v___x_411_, 1, v_a_407_);
return v___x_411_;
}
else
{
lean_object* v___x_412_; lean_object* v___x_413_; lean_object* v___x_414_; uint8_t v___x_415_; 
v___x_412_ = lean_unsigned_to_nat(0u);
v___x_413_ = l_Lean_Syntax_getArg(v_x_405_, v___x_412_);
v___x_414_ = ((lean_object*)(lp_mathlib_Functor___aux__Mathlib__Control__Functor______unexpand__Functor__mapConstRev__1___closed__1));
lean_inc(v___x_413_);
v___x_415_ = l_Lean_Syntax_isOfKind(v___x_413_, v___x_414_);
if (v___x_415_ == 0)
{
lean_object* v___x_416_; lean_object* v___x_417_; 
lean_dec(v___x_413_);
lean_dec(v_x_405_);
v___x_416_ = lean_box(0);
v___x_417_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_417_, 0, v___x_416_);
lean_ctor_set(v___x_417_, 1, v_a_407_);
return v___x_417_;
}
else
{
lean_object* v___x_418_; lean_object* v___x_419_; lean_object* v___x_420_; uint8_t v___x_421_; 
v___x_418_ = lean_unsigned_to_nat(1u);
v___x_419_ = l_Lean_Syntax_getArg(v_x_405_, v___x_418_);
lean_dec(v_x_405_);
v___x_420_ = lean_unsigned_to_nat(2u);
lean_inc(v___x_419_);
v___x_421_ = l_Lean_Syntax_matchesNull(v___x_419_, v___x_420_);
if (v___x_421_ == 0)
{
lean_object* v___x_422_; lean_object* v___x_423_; 
lean_dec(v___x_419_);
lean_dec(v___x_413_);
v___x_422_ = lean_box(0);
v___x_423_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_423_, 0, v___x_422_);
lean_ctor_set(v___x_423_, 1, v_a_407_);
return v___x_423_;
}
else
{
lean_object* v___x_424_; lean_object* v___x_425_; lean_object* v_ref_426_; uint8_t v___x_427_; lean_object* v___x_428_; lean_object* v___x_429_; lean_object* v___x_430_; lean_object* v___x_431_; lean_object* v___x_432_; lean_object* v___x_433_; 
v___x_424_ = l_Lean_Syntax_getArg(v___x_419_, v___x_412_);
v___x_425_ = l_Lean_Syntax_getArg(v___x_419_, v___x_418_);
lean_dec(v___x_419_);
v_ref_426_ = l_Lean_replaceRef(v___x_413_, v_a_406_);
lean_dec(v___x_413_);
v___x_427_ = 0;
v___x_428_ = l_Lean_SourceInfo_fromRef(v_ref_426_, v___x_427_);
lean_dec(v_ref_426_);
v___x_429_ = ((lean_object*)(lp_mathlib_Functor_term___x24_x3e___00__closed__2));
v___x_430_ = ((lean_object*)(lp_mathlib_Functor_term___x24_x3e___00__closed__5));
lean_inc(v___x_428_);
v___x_431_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_431_, 0, v___x_428_);
lean_ctor_set(v___x_431_, 1, v___x_430_);
v___x_432_ = l_Lean_Syntax_node3(v___x_428_, v___x_429_, v___x_424_, v___x_431_, v___x_425_);
v___x_433_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_433_, 0, v___x_432_);
lean_ctor_set(v___x_433_, 1, v_a_407_);
return v___x_433_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Functor___aux__Mathlib__Control__Functor______unexpand__Functor__mapConstRev__1___boxed(lean_object* v_x_434_, lean_object* v_a_435_, lean_object* v_a_436_){
_start:
{
lean_object* v_res_437_; 
v_res_437_ = lp_mathlib_Functor___aux__Mathlib__Control__Functor______unexpand__Functor__mapConstRev__1(v_x_434_, v_a_435_, v_a_436_);
lean_dec(v_a_435_);
return v_res_437_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Set_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Attr_Register(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Control_Functor(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Set_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Attr_Register(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Control_Functor(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Data_Set_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Attr_Register(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Control_Functor(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Set_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Attr_Register(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Control_Functor(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Control_Functor(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Control_Functor(builtin);
}
#ifdef __cplusplus
}
#endif
