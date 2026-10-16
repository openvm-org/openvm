// Lean compiler output
// Module: Mathlib.Tactic.TypeStar
// Imports: public import Init public meta import Init public import Mathlib.Tactic.Linter.Header
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
uint8_t lean_name_eq(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Term_getLevelNames___redArg(lean_object*);
lean_object* lean_name_append_index_after(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* l_Lean_Elab_Term_setLevelNames___redArg(lean_object*, lean_object*);
lean_object* l_Lean_mkLevelParam(lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
extern lean_object* l_Lean_Elab_unsupportedSyntaxExceptionId;
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Lean_Level_succ___override(lean_object*);
lean_object* l_Lean_Expr_sort___override(lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_List_elem___at___00__private_Mathlib_Tactic_TypeStar_0__Lean_Elab_Term_mkFreshLevelName_go_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_elem___at___00__private_Mathlib_Tactic_TypeStar_0__Lean_Elab_Term_mkFreshLevelName_go_spec__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_TypeStar_0__Lean_Elab_Term_mkFreshLevelName_go(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_TypeStar_0__Lean_Elab_Term_mkFreshLevelName_go___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Term_mkFreshLevelName(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Term_mkFreshLevelName___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Term_mkFreshLevelParam___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Term_mkFreshLevelParam___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Term_mkFreshLevelParam(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Term_mkFreshLevelParam___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_Elab_Term_termSort_x2a___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib_Lean_Elab_Term_termSort_x2a___closed__0 = (const lean_object*)&lp_mathlib_Lean_Elab_Term_termSort_x2a___closed__0_value;
static const lean_string_object lp_mathlib_Lean_Elab_Term_termSort_x2a___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Elab"};
static const lean_object* lp_mathlib_Lean_Elab_Term_termSort_x2a___closed__1 = (const lean_object*)&lp_mathlib_Lean_Elab_Term_termSort_x2a___closed__1_value;
static const lean_string_object lp_mathlib_Lean_Elab_Term_termSort_x2a___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_mathlib_Lean_Elab_Term_termSort_x2a___closed__2 = (const lean_object*)&lp_mathlib_Lean_Elab_Term_termSort_x2a___closed__2_value;
static const lean_string_object lp_mathlib_Lean_Elab_Term_termSort_x2a___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "termSort*"};
static const lean_object* lp_mathlib_Lean_Elab_Term_termSort_x2a___closed__3 = (const lean_object*)&lp_mathlib_Lean_Elab_Term_termSort_x2a___closed__3_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Term_termSort_x2a___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Elab_Term_termSort_x2a___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Lean_Elab_Term_termSort_x2a___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Term_termSort_x2a___closed__4_value_aux_0),((lean_object*)&lp_mathlib_Lean_Elab_Term_termSort_x2a___closed__1_value),LEAN_SCALAR_PTR_LITERAL(52, 247, 248, 201, 92, 23, 188, 159)}};
static const lean_ctor_object lp_mathlib_Lean_Elab_Term_termSort_x2a___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Term_termSort_x2a___closed__4_value_aux_1),((lean_object*)&lp_mathlib_Lean_Elab_Term_termSort_x2a___closed__2_value),LEAN_SCALAR_PTR_LITERAL(252, 225, 247, 249, 114, 131, 135, 109)}};
static const lean_ctor_object lp_mathlib_Lean_Elab_Term_termSort_x2a___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Term_termSort_x2a___closed__4_value_aux_2),((lean_object*)&lp_mathlib_Lean_Elab_Term_termSort_x2a___closed__3_value),LEAN_SCALAR_PTR_LITERAL(140, 247, 128, 6, 201, 106, 70, 133)}};
static const lean_object* lp_mathlib_Lean_Elab_Term_termSort_x2a___closed__4 = (const lean_object*)&lp_mathlib_Lean_Elab_Term_termSort_x2a___closed__4_value;
static const lean_string_object lp_mathlib_Lean_Elab_Term_termSort_x2a___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "Sort*"};
static const lean_object* lp_mathlib_Lean_Elab_Term_termSort_x2a___closed__5 = (const lean_object*)&lp_mathlib_Lean_Elab_Term_termSort_x2a___closed__5_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Term_termSort_x2a___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Term_termSort_x2a___closed__5_value)}};
static const lean_object* lp_mathlib_Lean_Elab_Term_termSort_x2a___closed__6 = (const lean_object*)&lp_mathlib_Lean_Elab_Term_termSort_x2a___closed__6_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Term_termSort_x2a___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Term_termSort_x2a___closed__4_value),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Elab_Term_termSort_x2a___closed__6_value)}};
static const lean_object* lp_mathlib_Lean_Elab_Term_termSort_x2a___closed__7 = (const lean_object*)&lp_mathlib_Lean_Elab_Term_termSort_x2a___closed__7_value;
LEAN_EXPORT const lean_object* lp_mathlib_Lean_Elab_Term_termSort_x2a = (const lean_object*)&lp_mathlib_Lean_Elab_Term_termSort_x2a___closed__7_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Term___aux__Mathlib__Tactic__TypeStar______elabRules__Lean__Elab__Term__termSort_x2a__1_spec__0___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Term___aux__Mathlib__Tactic__TypeStar______elabRules__Lean__Elab__Term__termSort_x2a__1_spec__0___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Term___aux__Mathlib__Tactic__TypeStar______elabRules__Lean__Elab__Term__termSort_x2a__1_spec__0___redArg();
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Term___aux__Mathlib__Tactic__TypeStar______elabRules__Lean__Elab__Term__termSort_x2a__1_spec__0___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Term___aux__Mathlib__Tactic__TypeStar______elabRules__Lean__Elab__Term__termSort_x2a__1_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Term___aux__Mathlib__Tactic__TypeStar______elabRules__Lean__Elab__Term__termSort_x2a__1_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Term___aux__Mathlib__Tactic__TypeStar______elabRules__Lean__Elab__Term__termSort_x2a__1___redArg___lam__0(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Lean_Elab_Term___aux__Mathlib__Tactic__TypeStar______elabRules__Lean__Elab__Term__termSort_x2a__1___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Lean_Elab_Term___aux__Mathlib__Tactic__TypeStar______elabRules__Lean__Elab__Term__termSort_x2a__1___redArg___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Lean_Elab_Term___aux__Mathlib__Tactic__TypeStar______elabRules__Lean__Elab__Term__termSort_x2a__1___redArg___closed__0 = (const lean_object*)&lp_mathlib_Lean_Elab_Term___aux__Mathlib__Tactic__TypeStar______elabRules__Lean__Elab__Term__termSort_x2a__1___redArg___closed__0_value;
static const lean_string_object lp_mathlib_Lean_Elab_Term___aux__Mathlib__Tactic__TypeStar______elabRules__Lean__Elab__Term__termSort_x2a__1___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "u"};
static const lean_object* lp_mathlib_Lean_Elab_Term___aux__Mathlib__Tactic__TypeStar______elabRules__Lean__Elab__Term__termSort_x2a__1___redArg___closed__1 = (const lean_object*)&lp_mathlib_Lean_Elab_Term___aux__Mathlib__Tactic__TypeStar______elabRules__Lean__Elab__Term__termSort_x2a__1___redArg___closed__1_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Term___aux__Mathlib__Tactic__TypeStar______elabRules__Lean__Elab__Term__termSort_x2a__1___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Elab_Term___aux__Mathlib__Tactic__TypeStar______elabRules__Lean__Elab__Term__termSort_x2a__1___redArg___closed__1_value),LEAN_SCALAR_PTR_LITERAL(232, 178, 247, 241, 102, 42, 87, 174)}};
static const lean_object* lp_mathlib_Lean_Elab_Term___aux__Mathlib__Tactic__TypeStar______elabRules__Lean__Elab__Term__termSort_x2a__1___redArg___closed__2 = (const lean_object*)&lp_mathlib_Lean_Elab_Term___aux__Mathlib__Tactic__TypeStar______elabRules__Lean__Elab__Term__termSort_x2a__1___redArg___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Term___aux__Mathlib__Tactic__TypeStar______elabRules__Lean__Elab__Term__termSort_x2a__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Term___aux__Mathlib__Tactic__TypeStar______elabRules__Lean__Elab__Term__termSort_x2a__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Term___aux__Mathlib__Tactic__TypeStar______elabRules__Lean__Elab__Term__termSort_x2a__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Term___aux__Mathlib__Tactic__TypeStar______elabRules__Lean__Elab__Term__termSort_x2a__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_Elab_Term_termType_x2a___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "termType*"};
static const lean_object* lp_mathlib_Lean_Elab_Term_termType_x2a___closed__0 = (const lean_object*)&lp_mathlib_Lean_Elab_Term_termType_x2a___closed__0_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Term_termType_x2a___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Elab_Term_termSort_x2a___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Lean_Elab_Term_termType_x2a___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Term_termType_x2a___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Lean_Elab_Term_termSort_x2a___closed__1_value),LEAN_SCALAR_PTR_LITERAL(52, 247, 248, 201, 92, 23, 188, 159)}};
static const lean_ctor_object lp_mathlib_Lean_Elab_Term_termType_x2a___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Term_termType_x2a___closed__1_value_aux_1),((lean_object*)&lp_mathlib_Lean_Elab_Term_termSort_x2a___closed__2_value),LEAN_SCALAR_PTR_LITERAL(252, 225, 247, 249, 114, 131, 135, 109)}};
static const lean_ctor_object lp_mathlib_Lean_Elab_Term_termType_x2a___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Term_termType_x2a___closed__1_value_aux_2),((lean_object*)&lp_mathlib_Lean_Elab_Term_termType_x2a___closed__0_value),LEAN_SCALAR_PTR_LITERAL(198, 85, 130, 159, 157, 244, 140, 188)}};
static const lean_object* lp_mathlib_Lean_Elab_Term_termType_x2a___closed__1 = (const lean_object*)&lp_mathlib_Lean_Elab_Term_termType_x2a___closed__1_value;
static const lean_string_object lp_mathlib_Lean_Elab_Term_termType_x2a___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "Type*"};
static const lean_object* lp_mathlib_Lean_Elab_Term_termType_x2a___closed__2 = (const lean_object*)&lp_mathlib_Lean_Elab_Term_termType_x2a___closed__2_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Term_termType_x2a___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Term_termType_x2a___closed__2_value)}};
static const lean_object* lp_mathlib_Lean_Elab_Term_termType_x2a___closed__3 = (const lean_object*)&lp_mathlib_Lean_Elab_Term_termType_x2a___closed__3_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Term_termType_x2a___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Term_termType_x2a___closed__1_value),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Elab_Term_termType_x2a___closed__3_value)}};
static const lean_object* lp_mathlib_Lean_Elab_Term_termType_x2a___closed__4 = (const lean_object*)&lp_mathlib_Lean_Elab_Term_termType_x2a___closed__4_value;
LEAN_EXPORT const lean_object* lp_mathlib_Lean_Elab_Term_termType_x2a = (const lean_object*)&lp_mathlib_Lean_Elab_Term_termType_x2a___closed__4_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Term___aux__Mathlib__Tactic__TypeStar______elabRules__Lean__Elab__Term__termType_x2a__1___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Term___aux__Mathlib__Tactic__TypeStar______elabRules__Lean__Elab__Term__termType_x2a__1___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Term___aux__Mathlib__Tactic__TypeStar______elabRules__Lean__Elab__Term__termType_x2a__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Term___aux__Mathlib__Tactic__TypeStar______elabRules__Lean__Elab__Term__termType_x2a__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_List_elem___at___00__private_Mathlib_Tactic_TypeStar_0__Lean_Elab_Term_mkFreshLevelName_go_spec__0(lean_object* v_a_1_, lean_object* v_x_2_){
_start:
{
if (lean_obj_tag(v_x_2_) == 0)
{
uint8_t v___x_3_; 
v___x_3_ = 0;
return v___x_3_;
}
else
{
lean_object* v_head_4_; lean_object* v_tail_5_; uint8_t v___x_6_; 
v_head_4_ = lean_ctor_get(v_x_2_, 0);
v_tail_5_ = lean_ctor_get(v_x_2_, 1);
v___x_6_ = lean_name_eq(v_a_1_, v_head_4_);
if (v___x_6_ == 0)
{
v_x_2_ = v_tail_5_;
goto _start;
}
else
{
return v___x_6_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_elem___at___00__private_Mathlib_Tactic_TypeStar_0__Lean_Elab_Term_mkFreshLevelName_go_spec__0___boxed(lean_object* v_a_8_, lean_object* v_x_9_){
_start:
{
uint8_t v_res_10_; lean_object* v_r_11_; 
v_res_10_ = lp_mathlib_List_elem___at___00__private_Mathlib_Tactic_TypeStar_0__Lean_Elab_Term_mkFreshLevelName_go_spec__0(v_a_8_, v_x_9_);
lean_dec(v_x_9_);
lean_dec(v_a_8_);
v_r_11_ = lean_box(v_res_10_);
return v_r_11_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_TypeStar_0__Lean_Elab_Term_mkFreshLevelName_go(lean_object* v_usedLevelNames_12_, lean_object* v_namePrefix_13_, lean_object* v_n_14_){
_start:
{
lean_object* v_u_15_; uint8_t v___x_16_; 
lean_inc(v_n_14_);
lean_inc(v_namePrefix_13_);
v_u_15_ = lean_name_append_index_after(v_namePrefix_13_, v_n_14_);
v___x_16_ = lp_mathlib_List_elem___at___00__private_Mathlib_Tactic_TypeStar_0__Lean_Elab_Term_mkFreshLevelName_go_spec__0(v_u_15_, v_usedLevelNames_12_);
if (v___x_16_ == 0)
{
lean_dec(v_n_14_);
lean_dec(v_namePrefix_13_);
return v_u_15_;
}
else
{
lean_object* v___x_17_; lean_object* v___x_18_; 
lean_dec(v_u_15_);
v___x_17_ = lean_unsigned_to_nat(1u);
v___x_18_ = lean_nat_add(v_n_14_, v___x_17_);
lean_dec(v_n_14_);
v_n_14_ = v___x_18_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_TypeStar_0__Lean_Elab_Term_mkFreshLevelName_go___boxed(lean_object* v_usedLevelNames_20_, lean_object* v_namePrefix_21_, lean_object* v_n_22_){
_start:
{
lean_object* v_res_23_; 
v_res_23_ = lp_mathlib___private_Mathlib_Tactic_TypeStar_0__Lean_Elab_Term_mkFreshLevelName_go(v_usedLevelNames_20_, v_namePrefix_21_, v_n_22_);
lean_dec(v_usedLevelNames_20_);
return v_res_23_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Term_mkFreshLevelName(lean_object* v_usedLevelNames_24_, lean_object* v_namePrefix_25_){
_start:
{
lean_object* v___x_26_; lean_object* v___x_27_; 
v___x_26_ = lean_unsigned_to_nat(1u);
v___x_27_ = lp_mathlib___private_Mathlib_Tactic_TypeStar_0__Lean_Elab_Term_mkFreshLevelName_go(v_usedLevelNames_24_, v_namePrefix_25_, v___x_26_);
return v___x_27_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Term_mkFreshLevelName___boxed(lean_object* v_usedLevelNames_28_, lean_object* v_namePrefix_29_){
_start:
{
lean_object* v_res_30_; 
v_res_30_ = lp_mathlib_Lean_Elab_Term_mkFreshLevelName(v_usedLevelNames_28_, v_namePrefix_29_);
lean_dec(v_usedLevelNames_28_);
return v_res_30_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Term_mkFreshLevelParam___redArg(lean_object* v_namePrefix_31_, lean_object* v_insert_32_, lean_object* v_a_33_){
_start:
{
lean_object* v___x_35_; 
v___x_35_ = l_Lean_Elab_Term_getLevelNames___redArg(v_a_33_);
if (lean_obj_tag(v___x_35_) == 0)
{
lean_object* v_a_36_; lean_object* v___x_37_; lean_object* v___x_38_; lean_object* v___x_39_; 
v_a_36_ = lean_ctor_get(v___x_35_, 0);
lean_inc(v_a_36_);
lean_dec_ref_known(v___x_35_, 1);
v___x_37_ = lp_mathlib_Lean_Elab_Term_mkFreshLevelName(v_a_36_, v_namePrefix_31_);
lean_inc(v___x_37_);
v___x_38_ = lean_apply_2(v_insert_32_, v_a_36_, v___x_37_);
v___x_39_ = l_Lean_Elab_Term_setLevelNames___redArg(v___x_38_, v_a_33_);
if (lean_obj_tag(v___x_39_) == 0)
{
lean_object* v___x_41_; uint8_t v_isShared_42_; uint8_t v_isSharedCheck_47_; 
v_isSharedCheck_47_ = !lean_is_exclusive(v___x_39_);
if (v_isSharedCheck_47_ == 0)
{
lean_object* v_unused_48_; 
v_unused_48_ = lean_ctor_get(v___x_39_, 0);
lean_dec(v_unused_48_);
v___x_41_ = v___x_39_;
v_isShared_42_ = v_isSharedCheck_47_;
goto v_resetjp_40_;
}
else
{
lean_dec(v___x_39_);
v___x_41_ = lean_box(0);
v_isShared_42_ = v_isSharedCheck_47_;
goto v_resetjp_40_;
}
v_resetjp_40_:
{
lean_object* v___x_43_; lean_object* v___x_45_; 
v___x_43_ = l_Lean_mkLevelParam(v___x_37_);
if (v_isShared_42_ == 0)
{
lean_ctor_set(v___x_41_, 0, v___x_43_);
v___x_45_ = v___x_41_;
goto v_reusejp_44_;
}
else
{
lean_object* v_reuseFailAlloc_46_; 
v_reuseFailAlloc_46_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_46_, 0, v___x_43_);
v___x_45_ = v_reuseFailAlloc_46_;
goto v_reusejp_44_;
}
v_reusejp_44_:
{
return v___x_45_;
}
}
}
else
{
lean_object* v_a_49_; lean_object* v___x_51_; uint8_t v_isShared_52_; uint8_t v_isSharedCheck_56_; 
lean_dec(v___x_37_);
v_a_49_ = lean_ctor_get(v___x_39_, 0);
v_isSharedCheck_56_ = !lean_is_exclusive(v___x_39_);
if (v_isSharedCheck_56_ == 0)
{
v___x_51_ = v___x_39_;
v_isShared_52_ = v_isSharedCheck_56_;
goto v_resetjp_50_;
}
else
{
lean_inc(v_a_49_);
lean_dec(v___x_39_);
v___x_51_ = lean_box(0);
v_isShared_52_ = v_isSharedCheck_56_;
goto v_resetjp_50_;
}
v_resetjp_50_:
{
lean_object* v___x_54_; 
if (v_isShared_52_ == 0)
{
v___x_54_ = v___x_51_;
goto v_reusejp_53_;
}
else
{
lean_object* v_reuseFailAlloc_55_; 
v_reuseFailAlloc_55_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_55_, 0, v_a_49_);
v___x_54_ = v_reuseFailAlloc_55_;
goto v_reusejp_53_;
}
v_reusejp_53_:
{
return v___x_54_;
}
}
}
}
else
{
lean_object* v_a_57_; lean_object* v___x_59_; uint8_t v_isShared_60_; uint8_t v_isSharedCheck_64_; 
lean_dec_ref(v_insert_32_);
lean_dec(v_namePrefix_31_);
v_a_57_ = lean_ctor_get(v___x_35_, 0);
v_isSharedCheck_64_ = !lean_is_exclusive(v___x_35_);
if (v_isSharedCheck_64_ == 0)
{
v___x_59_ = v___x_35_;
v_isShared_60_ = v_isSharedCheck_64_;
goto v_resetjp_58_;
}
else
{
lean_inc(v_a_57_);
lean_dec(v___x_35_);
v___x_59_ = lean_box(0);
v_isShared_60_ = v_isSharedCheck_64_;
goto v_resetjp_58_;
}
v_resetjp_58_:
{
lean_object* v___x_62_; 
if (v_isShared_60_ == 0)
{
v___x_62_ = v___x_59_;
goto v_reusejp_61_;
}
else
{
lean_object* v_reuseFailAlloc_63_; 
v_reuseFailAlloc_63_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_63_, 0, v_a_57_);
v___x_62_ = v_reuseFailAlloc_63_;
goto v_reusejp_61_;
}
v_reusejp_61_:
{
return v___x_62_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Term_mkFreshLevelParam___redArg___boxed(lean_object* v_namePrefix_65_, lean_object* v_insert_66_, lean_object* v_a_67_, lean_object* v_a_68_){
_start:
{
lean_object* v_res_69_; 
v_res_69_ = lp_mathlib_Lean_Elab_Term_mkFreshLevelParam___redArg(v_namePrefix_65_, v_insert_66_, v_a_67_);
lean_dec(v_a_67_);
return v_res_69_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Term_mkFreshLevelParam(lean_object* v_namePrefix_70_, lean_object* v_insert_71_, lean_object* v_a_72_, lean_object* v_a_73_, lean_object* v_a_74_, lean_object* v_a_75_, lean_object* v_a_76_, lean_object* v_a_77_){
_start:
{
lean_object* v___x_79_; 
v___x_79_ = lp_mathlib_Lean_Elab_Term_mkFreshLevelParam___redArg(v_namePrefix_70_, v_insert_71_, v_a_73_);
return v___x_79_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Term_mkFreshLevelParam___boxed(lean_object* v_namePrefix_80_, lean_object* v_insert_81_, lean_object* v_a_82_, lean_object* v_a_83_, lean_object* v_a_84_, lean_object* v_a_85_, lean_object* v_a_86_, lean_object* v_a_87_, lean_object* v_a_88_){
_start:
{
lean_object* v_res_89_; 
v_res_89_ = lp_mathlib_Lean_Elab_Term_mkFreshLevelParam(v_namePrefix_80_, v_insert_81_, v_a_82_, v_a_83_, v_a_84_, v_a_85_, v_a_86_, v_a_87_);
lean_dec(v_a_87_);
lean_dec_ref(v_a_86_);
lean_dec(v_a_85_);
lean_dec_ref(v_a_84_);
lean_dec(v_a_83_);
lean_dec_ref(v_a_82_);
return v_res_89_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Term___aux__Mathlib__Tactic__TypeStar______elabRules__Lean__Elab__Term__termSort_x2a__1_spec__0___redArg___closed__0(void){
_start:
{
lean_object* v___x_107_; lean_object* v___x_108_; lean_object* v___x_109_; 
v___x_107_ = lean_box(0);
v___x_108_ = l_Lean_Elab_unsupportedSyntaxExceptionId;
v___x_109_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_109_, 0, v___x_108_);
lean_ctor_set(v___x_109_, 1, v___x_107_);
return v___x_109_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Term___aux__Mathlib__Tactic__TypeStar______elabRules__Lean__Elab__Term__termSort_x2a__1_spec__0___redArg(){
_start:
{
lean_object* v___x_111_; lean_object* v___x_112_; 
v___x_111_ = lean_obj_once(&lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Term___aux__Mathlib__Tactic__TypeStar______elabRules__Lean__Elab__Term__termSort_x2a__1_spec__0___redArg___closed__0, &lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Term___aux__Mathlib__Tactic__TypeStar______elabRules__Lean__Elab__Term__termSort_x2a__1_spec__0___redArg___closed__0_once, _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Term___aux__Mathlib__Tactic__TypeStar______elabRules__Lean__Elab__Term__termSort_x2a__1_spec__0___redArg___closed__0);
v___x_112_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_112_, 0, v___x_111_);
return v___x_112_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Term___aux__Mathlib__Tactic__TypeStar______elabRules__Lean__Elab__Term__termSort_x2a__1_spec__0___redArg___boxed(lean_object* v___y_113_){
_start:
{
lean_object* v_res_114_; 
v_res_114_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Term___aux__Mathlib__Tactic__TypeStar______elabRules__Lean__Elab__Term__termSort_x2a__1_spec__0___redArg();
return v_res_114_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Term___aux__Mathlib__Tactic__TypeStar______elabRules__Lean__Elab__Term__termSort_x2a__1_spec__0(lean_object* v_00_u03b1_115_, lean_object* v___y_116_, lean_object* v___y_117_, lean_object* v___y_118_, lean_object* v___y_119_, lean_object* v___y_120_, lean_object* v___y_121_){
_start:
{
lean_object* v___x_123_; 
v___x_123_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Term___aux__Mathlib__Tactic__TypeStar______elabRules__Lean__Elab__Term__termSort_x2a__1_spec__0___redArg();
return v___x_123_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Term___aux__Mathlib__Tactic__TypeStar______elabRules__Lean__Elab__Term__termSort_x2a__1_spec__0___boxed(lean_object* v_00_u03b1_124_, lean_object* v___y_125_, lean_object* v___y_126_, lean_object* v___y_127_, lean_object* v___y_128_, lean_object* v___y_129_, lean_object* v___y_130_, lean_object* v___y_131_){
_start:
{
lean_object* v_res_132_; 
v_res_132_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Term___aux__Mathlib__Tactic__TypeStar______elabRules__Lean__Elab__Term__termSort_x2a__1_spec__0(v_00_u03b1_124_, v___y_125_, v___y_126_, v___y_127_, v___y_128_, v___y_129_, v___y_130_);
lean_dec(v___y_130_);
lean_dec_ref(v___y_129_);
lean_dec(v___y_128_);
lean_dec_ref(v___y_127_);
lean_dec(v___y_126_);
lean_dec_ref(v___y_125_);
return v_res_132_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Term___aux__Mathlib__Tactic__TypeStar______elabRules__Lean__Elab__Term__termSort_x2a__1___redArg___lam__0(lean_object* v_x_133_, lean_object* v_head_134_){
_start:
{
lean_object* v___x_135_; 
v___x_135_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_135_, 0, v_head_134_);
lean_ctor_set(v___x_135_, 1, v_x_133_);
return v___x_135_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Term___aux__Mathlib__Tactic__TypeStar______elabRules__Lean__Elab__Term__termSort_x2a__1___redArg(lean_object* v_stx_140_, lean_object* v_a_141_, lean_object* v_a_142_, lean_object* v_a_143_, lean_object* v_a_144_, lean_object* v_a_145_, lean_object* v_a_146_){
_start:
{
lean_object* v___x_148_; uint8_t v___x_149_; 
v___x_148_ = ((lean_object*)(lp_mathlib_Lean_Elab_Term_termSort_x2a___closed__4));
v___x_149_ = l_Lean_Syntax_isOfKind(v_stx_140_, v___x_148_);
if (v___x_149_ == 0)
{
lean_object* v___x_150_; 
v___x_150_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Term___aux__Mathlib__Tactic__TypeStar______elabRules__Lean__Elab__Term__termSort_x2a__1_spec__0___redArg();
return v___x_150_;
}
else
{
lean_object* v___f_151_; lean_object* v___x_152_; lean_object* v___x_153_; 
v___f_151_ = ((lean_object*)(lp_mathlib_Lean_Elab_Term___aux__Mathlib__Tactic__TypeStar______elabRules__Lean__Elab__Term__termSort_x2a__1___redArg___closed__0));
v___x_152_ = ((lean_object*)(lp_mathlib_Lean_Elab_Term___aux__Mathlib__Tactic__TypeStar______elabRules__Lean__Elab__Term__termSort_x2a__1___redArg___closed__2));
v___x_153_ = lp_mathlib_Lean_Elab_Term_mkFreshLevelParam___redArg(v___x_152_, v___f_151_, v_a_142_);
if (lean_obj_tag(v___x_153_) == 0)
{
lean_object* v_a_154_; lean_object* v___x_156_; uint8_t v_isShared_157_; uint8_t v_isSharedCheck_162_; 
v_a_154_ = lean_ctor_get(v___x_153_, 0);
v_isSharedCheck_162_ = !lean_is_exclusive(v___x_153_);
if (v_isSharedCheck_162_ == 0)
{
v___x_156_ = v___x_153_;
v_isShared_157_ = v_isSharedCheck_162_;
goto v_resetjp_155_;
}
else
{
lean_inc(v_a_154_);
lean_dec(v___x_153_);
v___x_156_ = lean_box(0);
v_isShared_157_ = v_isSharedCheck_162_;
goto v_resetjp_155_;
}
v_resetjp_155_:
{
lean_object* v___x_158_; lean_object* v___x_160_; 
v___x_158_ = l_Lean_Expr_sort___override(v_a_154_);
if (v_isShared_157_ == 0)
{
lean_ctor_set(v___x_156_, 0, v___x_158_);
v___x_160_ = v___x_156_;
goto v_reusejp_159_;
}
else
{
lean_object* v_reuseFailAlloc_161_; 
v_reuseFailAlloc_161_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_161_, 0, v___x_158_);
v___x_160_ = v_reuseFailAlloc_161_;
goto v_reusejp_159_;
}
v_reusejp_159_:
{
return v___x_160_;
}
}
}
else
{
lean_object* v_a_163_; lean_object* v___x_165_; uint8_t v_isShared_166_; uint8_t v_isSharedCheck_170_; 
v_a_163_ = lean_ctor_get(v___x_153_, 0);
v_isSharedCheck_170_ = !lean_is_exclusive(v___x_153_);
if (v_isSharedCheck_170_ == 0)
{
v___x_165_ = v___x_153_;
v_isShared_166_ = v_isSharedCheck_170_;
goto v_resetjp_164_;
}
else
{
lean_inc(v_a_163_);
lean_dec(v___x_153_);
v___x_165_ = lean_box(0);
v_isShared_166_ = v_isSharedCheck_170_;
goto v_resetjp_164_;
}
v_resetjp_164_:
{
lean_object* v___x_168_; 
if (v_isShared_166_ == 0)
{
v___x_168_ = v___x_165_;
goto v_reusejp_167_;
}
else
{
lean_object* v_reuseFailAlloc_169_; 
v_reuseFailAlloc_169_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_169_, 0, v_a_163_);
v___x_168_ = v_reuseFailAlloc_169_;
goto v_reusejp_167_;
}
v_reusejp_167_:
{
return v___x_168_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Term___aux__Mathlib__Tactic__TypeStar______elabRules__Lean__Elab__Term__termSort_x2a__1___redArg___boxed(lean_object* v_stx_171_, lean_object* v_a_172_, lean_object* v_a_173_, lean_object* v_a_174_, lean_object* v_a_175_, lean_object* v_a_176_, lean_object* v_a_177_, lean_object* v_a_178_){
_start:
{
lean_object* v_res_179_; 
v_res_179_ = lp_mathlib_Lean_Elab_Term___aux__Mathlib__Tactic__TypeStar______elabRules__Lean__Elab__Term__termSort_x2a__1___redArg(v_stx_171_, v_a_172_, v_a_173_, v_a_174_, v_a_175_, v_a_176_, v_a_177_);
lean_dec(v_a_177_);
lean_dec_ref(v_a_176_);
lean_dec(v_a_175_);
lean_dec_ref(v_a_174_);
lean_dec(v_a_173_);
lean_dec_ref(v_a_172_);
return v_res_179_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Term___aux__Mathlib__Tactic__TypeStar______elabRules__Lean__Elab__Term__termSort_x2a__1(lean_object* v_stx_180_, lean_object* v_x_181_, lean_object* v_a_182_, lean_object* v_a_183_, lean_object* v_a_184_, lean_object* v_a_185_, lean_object* v_a_186_, lean_object* v_a_187_){
_start:
{
lean_object* v___x_189_; 
v___x_189_ = lp_mathlib_Lean_Elab_Term___aux__Mathlib__Tactic__TypeStar______elabRules__Lean__Elab__Term__termSort_x2a__1___redArg(v_stx_180_, v_a_182_, v_a_183_, v_a_184_, v_a_185_, v_a_186_, v_a_187_);
return v___x_189_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Term___aux__Mathlib__Tactic__TypeStar______elabRules__Lean__Elab__Term__termSort_x2a__1___boxed(lean_object* v_stx_190_, lean_object* v_x_191_, lean_object* v_a_192_, lean_object* v_a_193_, lean_object* v_a_194_, lean_object* v_a_195_, lean_object* v_a_196_, lean_object* v_a_197_, lean_object* v_a_198_){
_start:
{
lean_object* v_res_199_; 
v_res_199_ = lp_mathlib_Lean_Elab_Term___aux__Mathlib__Tactic__TypeStar______elabRules__Lean__Elab__Term__termSort_x2a__1(v_stx_190_, v_x_191_, v_a_192_, v_a_193_, v_a_194_, v_a_195_, v_a_196_, v_a_197_);
lean_dec(v_a_197_);
lean_dec_ref(v_a_196_);
lean_dec(v_a_195_);
lean_dec_ref(v_a_194_);
lean_dec(v_a_193_);
lean_dec_ref(v_a_192_);
lean_dec(v_x_191_);
return v_res_199_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Term___aux__Mathlib__Tactic__TypeStar______elabRules__Lean__Elab__Term__termType_x2a__1___redArg(lean_object* v_stx_214_, lean_object* v_a_215_){
_start:
{
lean_object* v___x_217_; uint8_t v___x_218_; 
v___x_217_ = ((lean_object*)(lp_mathlib_Lean_Elab_Term_termType_x2a___closed__1));
v___x_218_ = l_Lean_Syntax_isOfKind(v_stx_214_, v___x_217_);
if (v___x_218_ == 0)
{
lean_object* v___x_219_; 
v___x_219_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Lean_Elab_Term___aux__Mathlib__Tactic__TypeStar______elabRules__Lean__Elab__Term__termSort_x2a__1_spec__0___redArg();
return v___x_219_;
}
else
{
lean_object* v___f_220_; lean_object* v___x_221_; lean_object* v___x_222_; 
v___f_220_ = ((lean_object*)(lp_mathlib_Lean_Elab_Term___aux__Mathlib__Tactic__TypeStar______elabRules__Lean__Elab__Term__termSort_x2a__1___redArg___closed__0));
v___x_221_ = ((lean_object*)(lp_mathlib_Lean_Elab_Term___aux__Mathlib__Tactic__TypeStar______elabRules__Lean__Elab__Term__termSort_x2a__1___redArg___closed__2));
v___x_222_ = lp_mathlib_Lean_Elab_Term_mkFreshLevelParam___redArg(v___x_221_, v___f_220_, v_a_215_);
if (lean_obj_tag(v___x_222_) == 0)
{
lean_object* v_a_223_; lean_object* v___x_225_; uint8_t v_isShared_226_; uint8_t v_isSharedCheck_232_; 
v_a_223_ = lean_ctor_get(v___x_222_, 0);
v_isSharedCheck_232_ = !lean_is_exclusive(v___x_222_);
if (v_isSharedCheck_232_ == 0)
{
v___x_225_ = v___x_222_;
v_isShared_226_ = v_isSharedCheck_232_;
goto v_resetjp_224_;
}
else
{
lean_inc(v_a_223_);
lean_dec(v___x_222_);
v___x_225_ = lean_box(0);
v_isShared_226_ = v_isSharedCheck_232_;
goto v_resetjp_224_;
}
v_resetjp_224_:
{
lean_object* v___x_227_; lean_object* v___x_228_; lean_object* v___x_230_; 
v___x_227_ = l_Lean_Level_succ___override(v_a_223_);
v___x_228_ = l_Lean_Expr_sort___override(v___x_227_);
if (v_isShared_226_ == 0)
{
lean_ctor_set(v___x_225_, 0, v___x_228_);
v___x_230_ = v___x_225_;
goto v_reusejp_229_;
}
else
{
lean_object* v_reuseFailAlloc_231_; 
v_reuseFailAlloc_231_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_231_, 0, v___x_228_);
v___x_230_ = v_reuseFailAlloc_231_;
goto v_reusejp_229_;
}
v_reusejp_229_:
{
return v___x_230_;
}
}
}
else
{
lean_object* v_a_233_; lean_object* v___x_235_; uint8_t v_isShared_236_; uint8_t v_isSharedCheck_240_; 
v_a_233_ = lean_ctor_get(v___x_222_, 0);
v_isSharedCheck_240_ = !lean_is_exclusive(v___x_222_);
if (v_isSharedCheck_240_ == 0)
{
v___x_235_ = v___x_222_;
v_isShared_236_ = v_isSharedCheck_240_;
goto v_resetjp_234_;
}
else
{
lean_inc(v_a_233_);
lean_dec(v___x_222_);
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
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Term___aux__Mathlib__Tactic__TypeStar______elabRules__Lean__Elab__Term__termType_x2a__1___redArg___boxed(lean_object* v_stx_241_, lean_object* v_a_242_, lean_object* v_a_243_){
_start:
{
lean_object* v_res_244_; 
v_res_244_ = lp_mathlib_Lean_Elab_Term___aux__Mathlib__Tactic__TypeStar______elabRules__Lean__Elab__Term__termType_x2a__1___redArg(v_stx_241_, v_a_242_);
lean_dec(v_a_242_);
return v_res_244_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Term___aux__Mathlib__Tactic__TypeStar______elabRules__Lean__Elab__Term__termType_x2a__1(lean_object* v_stx_245_, lean_object* v_x_246_, lean_object* v_a_247_, lean_object* v_a_248_, lean_object* v_a_249_, lean_object* v_a_250_, lean_object* v_a_251_, lean_object* v_a_252_){
_start:
{
lean_object* v___x_254_; 
v___x_254_ = lp_mathlib_Lean_Elab_Term___aux__Mathlib__Tactic__TypeStar______elabRules__Lean__Elab__Term__termType_x2a__1___redArg(v_stx_245_, v_a_248_);
return v___x_254_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Term___aux__Mathlib__Tactic__TypeStar______elabRules__Lean__Elab__Term__termType_x2a__1___boxed(lean_object* v_stx_255_, lean_object* v_x_256_, lean_object* v_a_257_, lean_object* v_a_258_, lean_object* v_a_259_, lean_object* v_a_260_, lean_object* v_a_261_, lean_object* v_a_262_, lean_object* v_a_263_){
_start:
{
lean_object* v_res_264_; 
v_res_264_ = lp_mathlib_Lean_Elab_Term___aux__Mathlib__Tactic__TypeStar______elabRules__Lean__Elab__Term__termType_x2a__1(v_stx_255_, v_x_256_, v_a_257_, v_a_258_, v_a_259_, v_a_260_, v_a_261_, v_a_262_);
lean_dec(v_a_262_);
lean_dec_ref(v_a_261_);
lean_dec(v_a_260_);
lean_dec_ref(v_a_259_);
lean_dec(v_a_258_);
lean_dec_ref(v_a_257_);
lean_dec(v_x_256_);
return v_res_264_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Linter_Header(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Tactic_TypeStar(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Linter_Header(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Tactic_TypeStar(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Tactic_Linter_Header(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_TypeStar(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Linter_Header(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_TypeStar(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Tactic_TypeStar(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Tactic_TypeStar(builtin);
}
#ifdef __cplusplus
}
#endif
