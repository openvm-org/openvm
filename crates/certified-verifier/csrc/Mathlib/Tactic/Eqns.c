// Lean compiler output
// Module: Mathlib.Tactic.Eqns
// Imports: public import Init public meta import Init public import Mathlib.Init public meta import Lean.Meta.Eqns public meta import Batteries.Lean.NameMapAttribute public meta import Lean.Elab.Exception public meta import Lean.Elab.InfoTree.Main
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
extern lean_object* l_Lean_Elab_unsupportedSyntaxExceptionId;
lean_object* lean_st_ref_get(lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArgs(lean_object*);
size_t lean_array_size(lean_object*);
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* lean_array_uget(lean_object*, size_t);
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
size_t lean_usize_add(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
lean_object* l_Lean_Elab_realizeGlobalConstNoOverloadWithInfo(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_batteries_Lean_registerNameMapAttribute___redArg(lean_object*);
lean_object* lp_batteries_Lean_NameMapExtension_find_x3f___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_registerGetEqnsFn(lean_object*);
static const lean_string_object lp_mathlib_eqns___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "eqns"};
static const lean_object* lp_mathlib_eqns___closed__0 = (const lean_object*)&lp_mathlib_eqns___closed__0_value;
static const lean_ctor_object lp_mathlib_eqns___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_eqns___closed__0_value),LEAN_SCALAR_PTR_LITERAL(189, 205, 217, 20, 6, 134, 86, 247)}};
static const lean_object* lp_mathlib_eqns___closed__1 = (const lean_object*)&lp_mathlib_eqns___closed__1_value;
static const lean_string_object lp_mathlib_eqns___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_eqns___closed__2 = (const lean_object*)&lp_mathlib_eqns___closed__2_value;
static const lean_ctor_object lp_mathlib_eqns___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_eqns___closed__2_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_eqns___closed__3 = (const lean_object*)&lp_mathlib_eqns___closed__3_value;
static const lean_ctor_object lp_mathlib_eqns___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib_eqns___closed__0_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_eqns___closed__4 = (const lean_object*)&lp_mathlib_eqns___closed__4_value;
static const lean_string_object lp_mathlib_eqns___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "many"};
static const lean_object* lp_mathlib_eqns___closed__5 = (const lean_object*)&lp_mathlib_eqns___closed__5_value;
static const lean_ctor_object lp_mathlib_eqns___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_eqns___closed__5_value),LEAN_SCALAR_PTR_LITERAL(41, 35, 40, 86, 189, 97, 244, 31)}};
static const lean_object* lp_mathlib_eqns___closed__6 = (const lean_object*)&lp_mathlib_eqns___closed__6_value;
static const lean_string_object lp_mathlib_eqns___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "ppSpace"};
static const lean_object* lp_mathlib_eqns___closed__7 = (const lean_object*)&lp_mathlib_eqns___closed__7_value;
static const lean_ctor_object lp_mathlib_eqns___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_eqns___closed__7_value),LEAN_SCALAR_PTR_LITERAL(207, 47, 58, 43, 30, 240, 125, 246)}};
static const lean_object* lp_mathlib_eqns___closed__8 = (const lean_object*)&lp_mathlib_eqns___closed__8_value;
static const lean_ctor_object lp_mathlib_eqns___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_eqns___closed__8_value)}};
static const lean_object* lp_mathlib_eqns___closed__9 = (const lean_object*)&lp_mathlib_eqns___closed__9_value;
static const lean_string_object lp_mathlib_eqns___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ident"};
static const lean_object* lp_mathlib_eqns___closed__10 = (const lean_object*)&lp_mathlib_eqns___closed__10_value;
static const lean_ctor_object lp_mathlib_eqns___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_eqns___closed__10_value),LEAN_SCALAR_PTR_LITERAL(52, 159, 208, 51, 14, 60, 6, 71)}};
static const lean_object* lp_mathlib_eqns___closed__11 = (const lean_object*)&lp_mathlib_eqns___closed__11_value;
static const lean_ctor_object lp_mathlib_eqns___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_eqns___closed__11_value)}};
static const lean_object* lp_mathlib_eqns___closed__12 = (const lean_object*)&lp_mathlib_eqns___closed__12_value;
static const lean_ctor_object lp_mathlib_eqns___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_eqns___closed__3_value),((lean_object*)&lp_mathlib_eqns___closed__9_value),((lean_object*)&lp_mathlib_eqns___closed__12_value)}};
static const lean_object* lp_mathlib_eqns___closed__13 = (const lean_object*)&lp_mathlib_eqns___closed__13_value;
static const lean_ctor_object lp_mathlib_eqns___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_eqns___closed__6_value),((lean_object*)&lp_mathlib_eqns___closed__13_value)}};
static const lean_object* lp_mathlib_eqns___closed__14 = (const lean_object*)&lp_mathlib_eqns___closed__14_value;
static const lean_ctor_object lp_mathlib_eqns___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_eqns___closed__3_value),((lean_object*)&lp_mathlib_eqns___closed__4_value),((lean_object*)&lp_mathlib_eqns___closed__14_value)}};
static const lean_object* lp_mathlib_eqns___closed__15 = (const lean_object*)&lp_mathlib_eqns___closed__15_value;
static const lean_ctor_object lp_mathlib_eqns___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_eqns___closed__1_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_mathlib_eqns___closed__15_value)}};
static const lean_object* lp_mathlib_eqns___closed__16 = (const lean_object*)&lp_mathlib_eqns___closed__16_value;
LEAN_EXPORT const lean_object* lp_mathlib_eqns = (const lean_object*)&lp_mathlib_eqns___closed__16_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_Eqns_0__initFn_00___x40_Mathlib_Tactic_Eqns_714549421____hygCtx___hyg_2__spec__0___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_Eqns_0__initFn_00___x40_Mathlib_Tactic_Eqns_714549421____hygCtx___hyg_2__spec__0___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_Eqns_0__initFn_00___x40_Mathlib_Tactic_Eqns_714549421____hygCtx___hyg_2__spec__0___redArg();
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_Eqns_0__initFn_00___x40_Mathlib_Tactic_Eqns_714549421____hygCtx___hyg_2__spec__0___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_Eqns_0__initFn_00___x40_Mathlib_Tactic_Eqns_714549421____hygCtx___hyg_2__spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_Eqns_0__initFn_00___x40_Mathlib_Tactic_Eqns_714549421____hygCtx___hyg_2__spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_Eqns_0__initFn_00___x40_Mathlib_Tactic_Eqns_714549421____hygCtx___hyg_2__spec__1(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_Eqns_0__initFn_00___x40_Mathlib_Tactic_Eqns_714549421____hygCtx___hyg_2__spec__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_Eqns_0__initFn_00___x40_Mathlib_Tactic_Eqns_714549421____hygCtx___hyg_2__spec__2(size_t, size_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_Eqns_0__initFn_00___x40_Mathlib_Tactic_Eqns_714549421____hygCtx___hyg_2__spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Eqns_0__initFn___lam__0_00___x40_Mathlib_Tactic_Eqns_714549421____hygCtx___hyg_2_(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Eqns_0__initFn___lam__0_00___x40_Mathlib_Tactic_Eqns_714549421____hygCtx___hyg_2____boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_Eqns_0__initFn___closed__0_00___x40_Mathlib_Tactic_Eqns_714549421____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_Eqns_0__initFn___lam__0_00___x40_Mathlib_Tactic_Eqns_714549421____hygCtx___hyg_2____boxed, .m_arity = 6, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_mathlib_eqns___closed__1_value)} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Eqns_0__initFn___closed__0_00___x40_Mathlib_Tactic_Eqns_714549421____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Eqns_0__initFn___closed__0_00___x40_Mathlib_Tactic_Eqns_714549421____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Eqns_0__initFn___closed__1_00___x40_Mathlib_Tactic_Eqns_714549421____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "eqnsAttribute"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Eqns_0__initFn___closed__1_00___x40_Mathlib_Tactic_Eqns_714549421____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Eqns_0__initFn___closed__1_00___x40_Mathlib_Tactic_Eqns_714549421____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Eqns_0__initFn___closed__2_00___x40_Mathlib_Tactic_Eqns_714549421____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Eqns_0__initFn___closed__1_00___x40_Mathlib_Tactic_Eqns_714549421____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(105, 100, 26, 79, 247, 112, 125, 49)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Eqns_0__initFn___closed__2_00___x40_Mathlib_Tactic_Eqns_714549421____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Eqns_0__initFn___closed__2_00___x40_Mathlib_Tactic_Eqns_714549421____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Eqns_0__initFn___closed__3_00___x40_Mathlib_Tactic_Eqns_714549421____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 69, .m_capacity = 69, .m_length = 68, .m_data = "Overrides the equation lemmas for a declaration to the provided list"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Eqns_0__initFn___closed__3_00___x40_Mathlib_Tactic_Eqns_714549421____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Eqns_0__initFn___closed__3_00___x40_Mathlib_Tactic_Eqns_714549421____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Eqns_0__initFn___closed__4_00___x40_Mathlib_Tactic_Eqns_714549421____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_eqns___closed__1_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Eqns_0__initFn___closed__2_00___x40_Mathlib_Tactic_Eqns_714549421____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Eqns_0__initFn___closed__3_00___x40_Mathlib_Tactic_Eqns_714549421____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Eqns_0__initFn___closed__0_00___x40_Mathlib_Tactic_Eqns_714549421____hygCtx___hyg_2__value)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Eqns_0__initFn___closed__4_00___x40_Mathlib_Tactic_Eqns_714549421____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Eqns_0__initFn___closed__4_00___x40_Mathlib_Tactic_Eqns_714549421____hygCtx___hyg_2__value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Eqns_0__initFn_00___x40_Mathlib_Tactic_Eqns_714549421____hygCtx___hyg_2_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Eqns_0__initFn_00___x40_Mathlib_Tactic_Eqns_714549421____hygCtx___hyg_2____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_eqnsAttribute;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Eqns_0__initFn___lam__0_00___x40_Mathlib_Tactic_Eqns_1919312714____hygCtx___hyg_2_(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Eqns_0__initFn___lam__0_00___x40_Mathlib_Tactic_Eqns_1919312714____hygCtx___hyg_2____boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_Eqns_0__initFn___closed__0_00___x40_Mathlib_Tactic_Eqns_1919312714____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_Eqns_0__initFn___lam__0_00___x40_Mathlib_Tactic_Eqns_1919312714____hygCtx___hyg_2____boxed, .m_arity = 6, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Eqns_0__initFn___closed__0_00___x40_Mathlib_Tactic_Eqns_1919312714____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Eqns_0__initFn___closed__0_00___x40_Mathlib_Tactic_Eqns_1919312714____hygCtx___hyg_2__value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Eqns_0__initFn_00___x40_Mathlib_Tactic_Eqns_1919312714____hygCtx___hyg_2_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Eqns_0__initFn_00___x40_Mathlib_Tactic_Eqns_1919312714____hygCtx___hyg_2____boxed(lean_object*);
static lean_object* _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_Eqns_0__initFn_00___x40_Mathlib_Tactic_Eqns_714549421____hygCtx___hyg_2__spec__0___redArg___closed__0(void){
_start:
{
lean_object* v___x_39_; lean_object* v___x_40_; lean_object* v___x_41_; 
v___x_39_ = lean_box(0);
v___x_40_ = l_Lean_Elab_unsupportedSyntaxExceptionId;
v___x_41_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_41_, 0, v___x_40_);
lean_ctor_set(v___x_41_, 1, v___x_39_);
return v___x_41_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_Eqns_0__initFn_00___x40_Mathlib_Tactic_Eqns_714549421____hygCtx___hyg_2__spec__0___redArg(){
_start:
{
lean_object* v___x_43_; lean_object* v___x_44_; 
v___x_43_ = lean_obj_once(&lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_Eqns_0__initFn_00___x40_Mathlib_Tactic_Eqns_714549421____hygCtx___hyg_2__spec__0___redArg___closed__0, &lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_Eqns_0__initFn_00___x40_Mathlib_Tactic_Eqns_714549421____hygCtx___hyg_2__spec__0___redArg___closed__0_once, _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_Eqns_0__initFn_00___x40_Mathlib_Tactic_Eqns_714549421____hygCtx___hyg_2__spec__0___redArg___closed__0);
v___x_44_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_44_, 0, v___x_43_);
return v___x_44_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_Eqns_0__initFn_00___x40_Mathlib_Tactic_Eqns_714549421____hygCtx___hyg_2__spec__0___redArg___boxed(lean_object* v___y_45_){
_start:
{
lean_object* v_res_46_; 
v_res_46_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_Eqns_0__initFn_00___x40_Mathlib_Tactic_Eqns_714549421____hygCtx___hyg_2__spec__0___redArg();
return v_res_46_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_Eqns_0__initFn_00___x40_Mathlib_Tactic_Eqns_714549421____hygCtx___hyg_2__spec__0(lean_object* v_00_u03b1_47_, lean_object* v___y_48_, lean_object* v___y_49_){
_start:
{
lean_object* v___x_51_; 
v___x_51_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_Eqns_0__initFn_00___x40_Mathlib_Tactic_Eqns_714549421____hygCtx___hyg_2__spec__0___redArg();
return v___x_51_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_Eqns_0__initFn_00___x40_Mathlib_Tactic_Eqns_714549421____hygCtx___hyg_2__spec__0___boxed(lean_object* v_00_u03b1_52_, lean_object* v___y_53_, lean_object* v___y_54_, lean_object* v___y_55_){
_start:
{
lean_object* v_res_56_; 
v_res_56_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_Eqns_0__initFn_00___x40_Mathlib_Tactic_Eqns_714549421____hygCtx___hyg_2__spec__0(v_00_u03b1_52_, v___y_53_, v___y_54_);
lean_dec(v___y_54_);
lean_dec_ref(v___y_53_);
return v_res_56_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_Eqns_0__initFn_00___x40_Mathlib_Tactic_Eqns_714549421____hygCtx___hyg_2__spec__1(size_t v_sz_57_, size_t v_i_58_, lean_object* v_bs_59_){
_start:
{
uint8_t v___x_60_; 
v___x_60_ = lean_usize_dec_lt(v_i_58_, v_sz_57_);
if (v___x_60_ == 0)
{
lean_object* v___x_61_; 
v___x_61_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_61_, 0, v_bs_59_);
return v___x_61_;
}
else
{
lean_object* v_v_62_; lean_object* v___x_63_; lean_object* v_bs_x27_64_; size_t v___x_65_; size_t v___x_66_; lean_object* v___x_67_; 
v_v_62_ = lean_array_uget(v_bs_59_, v_i_58_);
v___x_63_ = lean_unsigned_to_nat(0u);
v_bs_x27_64_ = lean_array_uset(v_bs_59_, v_i_58_, v___x_63_);
v___x_65_ = ((size_t)1ULL);
v___x_66_ = lean_usize_add(v_i_58_, v___x_65_);
v___x_67_ = lean_array_uset(v_bs_x27_64_, v_i_58_, v_v_62_);
v_i_58_ = v___x_66_;
v_bs_59_ = v___x_67_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_Eqns_0__initFn_00___x40_Mathlib_Tactic_Eqns_714549421____hygCtx___hyg_2__spec__1___boxed(lean_object* v_sz_69_, lean_object* v_i_70_, lean_object* v_bs_71_){
_start:
{
size_t v_sz_boxed_72_; size_t v_i_boxed_73_; lean_object* v_res_74_; 
v_sz_boxed_72_ = lean_unbox_usize(v_sz_69_);
lean_dec(v_sz_69_);
v_i_boxed_73_ = lean_unbox_usize(v_i_70_);
lean_dec(v_i_70_);
v_res_74_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_Eqns_0__initFn_00___x40_Mathlib_Tactic_Eqns_714549421____hygCtx___hyg_2__spec__1(v_sz_boxed_72_, v_i_boxed_73_, v_bs_71_);
return v_res_74_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_Eqns_0__initFn_00___x40_Mathlib_Tactic_Eqns_714549421____hygCtx___hyg_2__spec__2(size_t v_sz_75_, size_t v_i_76_, lean_object* v_bs_77_, lean_object* v___y_78_, lean_object* v___y_79_){
_start:
{
uint8_t v___x_81_; 
v___x_81_ = lean_usize_dec_lt(v_i_76_, v_sz_75_);
if (v___x_81_ == 0)
{
lean_object* v___x_82_; 
v___x_82_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_82_, 0, v_bs_77_);
return v___x_82_;
}
else
{
lean_object* v_v_83_; lean_object* v___x_84_; lean_object* v___x_85_; 
v_v_83_ = lean_array_uget_borrowed(v_bs_77_, v_i_76_);
v___x_84_ = lean_box(0);
lean_inc(v_v_83_);
v___x_85_ = l_Lean_Elab_realizeGlobalConstNoOverloadWithInfo(v_v_83_, v___x_84_, v___y_78_, v___y_79_);
if (lean_obj_tag(v___x_85_) == 0)
{
lean_object* v_a_86_; lean_object* v___x_87_; lean_object* v_bs_x27_88_; size_t v___x_89_; size_t v___x_90_; lean_object* v___x_91_; 
v_a_86_ = lean_ctor_get(v___x_85_, 0);
lean_inc(v_a_86_);
lean_dec_ref_known(v___x_85_, 1);
v___x_87_ = lean_unsigned_to_nat(0u);
v_bs_x27_88_ = lean_array_uset(v_bs_77_, v_i_76_, v___x_87_);
v___x_89_ = ((size_t)1ULL);
v___x_90_ = lean_usize_add(v_i_76_, v___x_89_);
v___x_91_ = lean_array_uset(v_bs_x27_88_, v_i_76_, v_a_86_);
v_i_76_ = v___x_90_;
v_bs_77_ = v___x_91_;
goto _start;
}
else
{
lean_object* v_a_93_; lean_object* v___x_95_; uint8_t v_isShared_96_; uint8_t v_isSharedCheck_100_; 
lean_dec_ref(v_bs_77_);
v_a_93_ = lean_ctor_get(v___x_85_, 0);
v_isSharedCheck_100_ = !lean_is_exclusive(v___x_85_);
if (v_isSharedCheck_100_ == 0)
{
v___x_95_ = v___x_85_;
v_isShared_96_ = v_isSharedCheck_100_;
goto v_resetjp_94_;
}
else
{
lean_inc(v_a_93_);
lean_dec(v___x_85_);
v___x_95_ = lean_box(0);
v_isShared_96_ = v_isSharedCheck_100_;
goto v_resetjp_94_;
}
v_resetjp_94_:
{
lean_object* v___x_98_; 
if (v_isShared_96_ == 0)
{
v___x_98_ = v___x_95_;
goto v_reusejp_97_;
}
else
{
lean_object* v_reuseFailAlloc_99_; 
v_reuseFailAlloc_99_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_99_, 0, v_a_93_);
v___x_98_ = v_reuseFailAlloc_99_;
goto v_reusejp_97_;
}
v_reusejp_97_:
{
return v___x_98_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_Eqns_0__initFn_00___x40_Mathlib_Tactic_Eqns_714549421____hygCtx___hyg_2__spec__2___boxed(lean_object* v_sz_101_, lean_object* v_i_102_, lean_object* v_bs_103_, lean_object* v___y_104_, lean_object* v___y_105_, lean_object* v___y_106_){
_start:
{
size_t v_sz_boxed_107_; size_t v_i_boxed_108_; lean_object* v_res_109_; 
v_sz_boxed_107_ = lean_unbox_usize(v_sz_101_);
lean_dec(v_sz_101_);
v_i_boxed_108_ = lean_unbox_usize(v_i_102_);
lean_dec(v_i_102_);
v_res_109_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_Eqns_0__initFn_00___x40_Mathlib_Tactic_Eqns_714549421____hygCtx___hyg_2__spec__2(v_sz_boxed_107_, v_i_boxed_108_, v_bs_103_, v___y_104_, v___y_105_);
lean_dec(v___y_105_);
lean_dec_ref(v___y_104_);
return v_res_109_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Eqns_0__initFn___lam__0_00___x40_Mathlib_Tactic_Eqns_714549421____hygCtx___hyg_2_(lean_object* v___x_110_, lean_object* v_x_111_, lean_object* v_x_112_, lean_object* v___y_113_, lean_object* v___y_114_){
_start:
{
uint8_t v___x_116_; 
lean_inc(v_x_112_);
v___x_116_ = l_Lean_Syntax_isOfKind(v_x_112_, v___x_110_);
if (v___x_116_ == 0)
{
lean_object* v___x_117_; 
lean_dec(v_x_112_);
v___x_117_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_Eqns_0__initFn_00___x40_Mathlib_Tactic_Eqns_714549421____hygCtx___hyg_2__spec__0___redArg();
return v___x_117_;
}
else
{
lean_object* v___x_118_; lean_object* v___x_119_; lean_object* v___x_120_; size_t v_sz_121_; size_t v___x_122_; lean_object* v___x_123_; 
v___x_118_ = lean_unsigned_to_nat(1u);
v___x_119_ = l_Lean_Syntax_getArg(v_x_112_, v___x_118_);
lean_dec(v_x_112_);
v___x_120_ = l_Lean_Syntax_getArgs(v___x_119_);
lean_dec(v___x_119_);
v_sz_121_ = lean_array_size(v___x_120_);
v___x_122_ = ((size_t)0ULL);
v___x_123_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_Eqns_0__initFn_00___x40_Mathlib_Tactic_Eqns_714549421____hygCtx___hyg_2__spec__1(v_sz_121_, v___x_122_, v___x_120_);
if (lean_obj_tag(v___x_123_) == 0)
{
lean_object* v___x_124_; 
v___x_124_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_Eqns_0__initFn_00___x40_Mathlib_Tactic_Eqns_714549421____hygCtx___hyg_2__spec__0___redArg();
return v___x_124_;
}
else
{
lean_object* v_val_125_; size_t v_sz_126_; lean_object* v___x_127_; 
v_val_125_ = lean_ctor_get(v___x_123_, 0);
lean_inc(v_val_125_);
lean_dec_ref_known(v___x_123_, 1);
v_sz_126_ = lean_array_size(v_val_125_);
v___x_127_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_Eqns_0__initFn_00___x40_Mathlib_Tactic_Eqns_714549421____hygCtx___hyg_2__spec__2(v_sz_126_, v___x_122_, v_val_125_, v___y_113_, v___y_114_);
return v___x_127_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Eqns_0__initFn___lam__0_00___x40_Mathlib_Tactic_Eqns_714549421____hygCtx___hyg_2____boxed(lean_object* v___x_128_, lean_object* v_x_129_, lean_object* v_x_130_, lean_object* v___y_131_, lean_object* v___y_132_, lean_object* v___y_133_){
_start:
{
lean_object* v_res_134_; 
v_res_134_ = lp_mathlib___private_Mathlib_Tactic_Eqns_0__initFn___lam__0_00___x40_Mathlib_Tactic_Eqns_714549421____hygCtx___hyg_2_(v___x_128_, v_x_129_, v_x_130_, v___y_131_, v___y_132_);
lean_dec(v___y_132_);
lean_dec_ref(v___y_131_);
lean_dec(v_x_129_);
lean_dec(v___x_128_);
return v_res_134_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Eqns_0__initFn_00___x40_Mathlib_Tactic_Eqns_714549421____hygCtx___hyg_2_(){
_start:
{
lean_object* v___x_147_; lean_object* v___x_148_; 
v___x_147_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Eqns_0__initFn___closed__4_00___x40_Mathlib_Tactic_Eqns_714549421____hygCtx___hyg_2_));
v___x_148_ = lp_batteries_Lean_registerNameMapAttribute___redArg(v___x_147_);
return v___x_148_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Eqns_0__initFn_00___x40_Mathlib_Tactic_Eqns_714549421____hygCtx___hyg_2____boxed(lean_object* v_a_149_){
_start:
{
lean_object* v_res_150_; 
v_res_150_ = lp_mathlib___private_Mathlib_Tactic_Eqns_0__initFn_00___x40_Mathlib_Tactic_Eqns_714549421____hygCtx___hyg_2_();
return v_res_150_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Eqns_0__initFn___lam__0_00___x40_Mathlib_Tactic_Eqns_1919312714____hygCtx___hyg_2_(lean_object* v_name_151_, lean_object* v___y_152_, lean_object* v___y_153_, lean_object* v___y_154_, lean_object* v___y_155_){
_start:
{
lean_object* v___x_157_; lean_object* v_env_158_; lean_object* v___x_159_; lean_object* v___x_160_; lean_object* v___x_161_; 
v___x_157_ = lean_st_ref_get(v___y_155_);
v_env_158_ = lean_ctor_get(v___x_157_, 0);
lean_inc_ref(v_env_158_);
lean_dec(v___x_157_);
v___x_159_ = lp_mathlib_eqnsAttribute;
v___x_160_ = lp_batteries_Lean_NameMapExtension_find_x3f___redArg(v___x_159_, v_env_158_, v_name_151_);
v___x_161_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_161_, 0, v___x_160_);
return v___x_161_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Eqns_0__initFn___lam__0_00___x40_Mathlib_Tactic_Eqns_1919312714____hygCtx___hyg_2____boxed(lean_object* v_name_162_, lean_object* v___y_163_, lean_object* v___y_164_, lean_object* v___y_165_, lean_object* v___y_166_, lean_object* v___y_167_){
_start:
{
lean_object* v_res_168_; 
v_res_168_ = lp_mathlib___private_Mathlib_Tactic_Eqns_0__initFn___lam__0_00___x40_Mathlib_Tactic_Eqns_1919312714____hygCtx___hyg_2_(v_name_162_, v___y_163_, v___y_164_, v___y_165_, v___y_166_);
lean_dec(v___y_166_);
lean_dec_ref(v___y_165_);
lean_dec(v___y_164_);
lean_dec_ref(v___y_163_);
lean_dec(v_name_162_);
return v_res_168_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Eqns_0__initFn_00___x40_Mathlib_Tactic_Eqns_1919312714____hygCtx___hyg_2_(){
_start:
{
lean_object* v___f_171_; lean_object* v___x_172_; 
v___f_171_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Eqns_0__initFn___closed__0_00___x40_Mathlib_Tactic_Eqns_1919312714____hygCtx___hyg_2_));
v___x_172_ = l_Lean_Meta_registerGetEqnsFn(v___f_171_);
return v___x_172_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Eqns_0__initFn_00___x40_Mathlib_Tactic_Eqns_1919312714____hygCtx___hyg_2____boxed(lean_object* v_a_173_){
_start:
{
lean_object* v_res_174_; 
v_res_174_ = lp_mathlib___private_Mathlib_Tactic_Eqns_0__initFn_00___x40_Mathlib_Tactic_Eqns_1919312714____hygCtx___hyg_2_();
return v_res_174_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Init(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Eqns(uint8_t builtin) {
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
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_Lean_Meta_Eqns(uint8_t builtin);
lean_object* runtime_initialize_batteries_Batteries_Lean_NameMapAttribute(uint8_t builtin);
lean_object* runtime_initialize_Lean_Elab_Exception(uint8_t builtin);
lean_object* runtime_initialize_Lean_Elab_InfoTree_Main(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Tactic_Eqns(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Meta_Eqns(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Lean_NameMapAttribute(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Elab_Exception(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Elab_InfoTree_Main(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = lp_mathlib___private_Mathlib_Tactic_Eqns_0__initFn_00___x40_Mathlib_Tactic_Eqns_714549421____hygCtx___hyg_2_();
if (lean_io_result_is_error(res)) return res;
lp_mathlib_eqnsAttribute = lean_io_result_get_value(res);
lean_mark_persistent(lp_mathlib_eqnsAttribute);
lean_dec_ref(res);
res = lp_mathlib___private_Mathlib_Tactic_Eqns_0__initFn_00___x40_Mathlib_Tactic_Eqns_1919312714____hygCtx___hyg_2_();
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Init(uint8_t builtin);
lean_object* initialize_Lean_Meta_Eqns(uint8_t builtin);
lean_object* initialize_batteries_Batteries_Lean_NameMapAttribute(uint8_t builtin);
lean_object* initialize_Lean_Elab_Exception(uint8_t builtin);
lean_object* initialize_Lean_Elab_InfoTree_Main(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_Eqns(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Meta_Eqns(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_Lean_NameMapAttribute(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Elab_Exception(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Elab_InfoTree_Main(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Eqns(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Tactic_Eqns(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Tactic_Eqns(builtin);
}
#ifdef __cplusplus
}
#endif
