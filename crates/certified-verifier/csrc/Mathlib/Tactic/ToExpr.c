// Lean compiler output
// Module: Mathlib.Tactic.ToExpr
// Imports: public import Init public meta import Init public import Mathlib.Init
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
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* l_Lean_Expr_const___override(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr3(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_mkNatLit(lean_object*);
lean_object* l_Lean_Expr_app___override(lean_object*, lean_object*);
lean_object* l_Lean_mkConst(lean_object*, lean_object*);
lean_object* l___private_Lean_ToExpr_0__Lean_Name_toExprAux(lean_object*);
lean_object* l_Lean_mkAppB(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_lit___override(lean_object*);
lean_object* l_Lean_mkStrLit(lean_object*);
lean_object* l_Lean_mkApp3(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_nat_to_int(lean_object*);
uint8_t lean_int_dec_le(lean_object*, lean_object*);
lean_object* l_Lean_Level_ofNat(lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* lean_int_neg(lean_object*);
lean_object* l_Int_toNat(lean_object*);
lean_object* l_Lean_instToExprInt_mkNat(lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_to_list(lean_object*);
lean_object* l_Lean_Level_succ___override(lean_object*);
static const lean_string_object lp_mathlib_Mathlib_instToExprULift__mathlib_toExpr___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ULift"};
static const lean_object* lp_mathlib_Mathlib_instToExprULift__mathlib_toExpr___redArg___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_instToExprULift__mathlib_toExpr___redArg___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_instToExprULift__mathlib_toExpr___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "up"};
static const lean_object* lp_mathlib_Mathlib_instToExprULift__mathlib_toExpr___redArg___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_instToExprULift__mathlib_toExpr___redArg___closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_instToExprULift__mathlib_toExpr___redArg___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_instToExprULift__mathlib_toExpr___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(14, 162, 24, 1, 186, 170, 9, 57)}};
static const lean_ctor_object lp_mathlib_Mathlib_instToExprULift__mathlib_toExpr___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_instToExprULift__mathlib_toExpr___redArg___closed__2_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_instToExprULift__mathlib_toExpr___redArg___closed__1_value),LEAN_SCALAR_PTR_LITERAL(38, 138, 107, 231, 89, 86, 217, 216)}};
static const lean_object* lp_mathlib_Mathlib_instToExprULift__mathlib_toExpr___redArg___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_instToExprULift__mathlib_toExpr___redArg___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_instToExprULift__mathlib_toExpr___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_instToExprULift__mathlib_toExpr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib_Mathlib_instToExprULift__mathlib___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_instToExprULift__mathlib_toExpr___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(14, 162, 24, 1, 186, 170, 9, 57)}};
static const lean_object* lp_mathlib_Mathlib_instToExprULift__mathlib___redArg___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_instToExprULift__mathlib___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_instToExprULift__mathlib___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_instToExprULift__mathlib(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_instToExprPUnitOfToLevel__mathlib___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "PUnit"};
static const lean_object* lp_mathlib_Mathlib_instToExprPUnitOfToLevel__mathlib___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_instToExprPUnitOfToLevel__mathlib___lam__0___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_instToExprPUnitOfToLevel__mathlib___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "unit"};
static const lean_object* lp_mathlib_Mathlib_instToExprPUnitOfToLevel__mathlib___lam__0___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_instToExprPUnitOfToLevel__mathlib___lam__0___closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_instToExprPUnitOfToLevel__mathlib___lam__0___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_instToExprPUnitOfToLevel__mathlib___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(23, 153, 158, 141, 176, 162, 235, 153)}};
static const lean_ctor_object lp_mathlib_Mathlib_instToExprPUnitOfToLevel__mathlib___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_instToExprPUnitOfToLevel__mathlib___lam__0___closed__2_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_instToExprPUnitOfToLevel__mathlib___lam__0___closed__1_value),LEAN_SCALAR_PTR_LITERAL(146, 91, 82, 196, 249, 72, 203, 194)}};
static const lean_object* lp_mathlib_Mathlib_instToExprPUnitOfToLevel__mathlib___lam__0___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_instToExprPUnitOfToLevel__mathlib___lam__0___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_instToExprPUnitOfToLevel__mathlib___lam__0(lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib_Mathlib_instToExprPUnitOfToLevel__mathlib___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_instToExprPUnitOfToLevel__mathlib___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(23, 153, 158, 141, 176, 162, 235, 153)}};
static const lean_object* lp_mathlib_Mathlib_instToExprPUnitOfToLevel__mathlib___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_instToExprPUnitOfToLevel__mathlib___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_instToExprPUnitOfToLevel__mathlib(lean_object*);
static const lean_string_object lp_mathlib_Mathlib_instToExprRaw__mathlib_toExpr___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "String"};
static const lean_object* lp_mathlib_Mathlib_instToExprRaw__mathlib_toExpr___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_instToExprRaw__mathlib_toExpr___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_instToExprRaw__mathlib_toExpr___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Pos"};
static const lean_object* lp_mathlib_Mathlib_instToExprRaw__mathlib_toExpr___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_instToExprRaw__mathlib_toExpr___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_instToExprRaw__mathlib_toExpr___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Raw"};
static const lean_object* lp_mathlib_Mathlib_instToExprRaw__mathlib_toExpr___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_instToExprRaw__mathlib_toExpr___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_instToExprRaw__mathlib_toExpr___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "mk"};
static const lean_object* lp_mathlib_Mathlib_instToExprRaw__mathlib_toExpr___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_instToExprRaw__mathlib_toExpr___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_instToExprRaw__mathlib_toExpr___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_instToExprRaw__mathlib_toExpr___closed__0_value),LEAN_SCALAR_PTR_LITERAL(6, 130, 56, 8, 41, 104, 134, 43)}};
static const lean_ctor_object lp_mathlib_Mathlib_instToExprRaw__mathlib_toExpr___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_instToExprRaw__mathlib_toExpr___closed__4_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_instToExprRaw__mathlib_toExpr___closed__1_value),LEAN_SCALAR_PTR_LITERAL(207, 230, 80, 37, 136, 222, 125, 174)}};
static const lean_ctor_object lp_mathlib_Mathlib_instToExprRaw__mathlib_toExpr___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_instToExprRaw__mathlib_toExpr___closed__4_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_instToExprRaw__mathlib_toExpr___closed__2_value),LEAN_SCALAR_PTR_LITERAL(192, 160, 30, 114, 46, 165, 46, 109)}};
static const lean_ctor_object lp_mathlib_Mathlib_instToExprRaw__mathlib_toExpr___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_instToExprRaw__mathlib_toExpr___closed__4_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_instToExprRaw__mathlib_toExpr___closed__3_value),LEAN_SCALAR_PTR_LITERAL(184, 144, 176, 63, 14, 104, 245, 146)}};
static const lean_object* lp_mathlib_Mathlib_instToExprRaw__mathlib_toExpr___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_instToExprRaw__mathlib_toExpr___closed__4_value;
static lean_once_cell_t lp_mathlib_Mathlib_instToExprRaw__mathlib_toExpr___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_instToExprRaw__mathlib_toExpr___closed__5;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_instToExprRaw__mathlib_toExpr(lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_instToExprRaw__mathlib___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_instToExprRaw__mathlib_toExpr, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_instToExprRaw__mathlib___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_instToExprRaw__mathlib___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_instToExprRaw__mathlib___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_instToExprRaw__mathlib_toExpr___closed__0_value),LEAN_SCALAR_PTR_LITERAL(6, 130, 56, 8, 41, 104, 134, 43)}};
static const lean_ctor_object lp_mathlib_Mathlib_instToExprRaw__mathlib___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_instToExprRaw__mathlib___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_instToExprRaw__mathlib_toExpr___closed__1_value),LEAN_SCALAR_PTR_LITERAL(207, 230, 80, 37, 136, 222, 125, 174)}};
static const lean_ctor_object lp_mathlib_Mathlib_instToExprRaw__mathlib___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_instToExprRaw__mathlib___closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_instToExprRaw__mathlib_toExpr___closed__2_value),LEAN_SCALAR_PTR_LITERAL(192, 160, 30, 114, 46, 165, 46, 109)}};
static const lean_object* lp_mathlib_Mathlib_instToExprRaw__mathlib___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_instToExprRaw__mathlib___closed__1_value;
static lean_once_cell_t lp_mathlib_Mathlib_instToExprRaw__mathlib___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_instToExprRaw__mathlib___closed__2;
static lean_once_cell_t lp_mathlib_Mathlib_instToExprRaw__mathlib___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_instToExprRaw__mathlib___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_instToExprRaw__mathlib;
static const lean_string_object lp_mathlib_Mathlib_instToExprRaw__mathlib__1_toExpr___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "Substring"};
static const lean_object* lp_mathlib_Mathlib_instToExprRaw__mathlib__1_toExpr___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_instToExprRaw__mathlib__1_toExpr___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_instToExprRaw__mathlib__1_toExpr___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_instToExprRaw__mathlib__1_toExpr___closed__0_value),LEAN_SCALAR_PTR_LITERAL(253, 41, 71, 226, 169, 224, 163, 8)}};
static const lean_ctor_object lp_mathlib_Mathlib_instToExprRaw__mathlib__1_toExpr___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_instToExprRaw__mathlib__1_toExpr___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_instToExprRaw__mathlib_toExpr___closed__2_value),LEAN_SCALAR_PTR_LITERAL(218, 24, 183, 11, 177, 78, 166, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_instToExprRaw__mathlib__1_toExpr___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_instToExprRaw__mathlib__1_toExpr___closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_instToExprRaw__mathlib_toExpr___closed__3_value),LEAN_SCALAR_PTR_LITERAL(234, 69, 120, 41, 208, 100, 53, 41)}};
static const lean_object* lp_mathlib_Mathlib_instToExprRaw__mathlib__1_toExpr___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_instToExprRaw__mathlib__1_toExpr___closed__1_value;
static lean_once_cell_t lp_mathlib_Mathlib_instToExprRaw__mathlib__1_toExpr___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_instToExprRaw__mathlib__1_toExpr___closed__2;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_instToExprRaw__mathlib__1_toExpr(lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_instToExprRaw__mathlib__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_instToExprRaw__mathlib__1_toExpr, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_instToExprRaw__mathlib__1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_instToExprRaw__mathlib__1___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_instToExprRaw__mathlib__1___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_instToExprRaw__mathlib__1_toExpr___closed__0_value),LEAN_SCALAR_PTR_LITERAL(253, 41, 71, 226, 169, 224, 163, 8)}};
static const lean_ctor_object lp_mathlib_Mathlib_instToExprRaw__mathlib__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_instToExprRaw__mathlib__1___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_instToExprRaw__mathlib_toExpr___closed__2_value),LEAN_SCALAR_PTR_LITERAL(218, 24, 183, 11, 177, 78, 166, 111)}};
static const lean_object* lp_mathlib_Mathlib_instToExprRaw__mathlib__1___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_instToExprRaw__mathlib__1___closed__1_value;
static lean_once_cell_t lp_mathlib_Mathlib_instToExprRaw__mathlib__1___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_instToExprRaw__mathlib__1___closed__2;
static lean_once_cell_t lp_mathlib_Mathlib_instToExprRaw__mathlib__1___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_instToExprRaw__mathlib__1___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_instToExprRaw__mathlib__1;
static const lean_string_object lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "SourceInfo"};
static const lean_object* lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "original"};
static const lean_object* lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr___closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr___closed__3_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr___closed__1_value),LEAN_SCALAR_PTR_LITERAL(185, 145, 56, 44, 129, 38, 15, 45)}};
static const lean_ctor_object lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr___closed__3_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr___closed__2_value),LEAN_SCALAR_PTR_LITERAL(89, 39, 224, 222, 205, 193, 172, 25)}};
static const lean_object* lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr___closed__3_value;
static lean_once_cell_t lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr___closed__4;
static const lean_string_object lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "synthetic"};
static const lean_object* lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr___closed__5_value;
static const lean_ctor_object lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr___closed__6_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr___closed__6_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr___closed__6_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr___closed__1_value),LEAN_SCALAR_PTR_LITERAL(185, 145, 56, 44, 129, 38, 15, 45)}};
static const lean_ctor_object lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr___closed__6_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr___closed__5_value),LEAN_SCALAR_PTR_LITERAL(252, 175, 237, 151, 118, 97, 132, 97)}};
static const lean_object* lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr___closed__6_value;
static lean_once_cell_t lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr___closed__7;
static const lean_string_object lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Bool"};
static const lean_object* lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr___closed__8_value;
static const lean_string_object lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "false"};
static const lean_object* lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr___closed__9_value;
static const lean_ctor_object lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr___closed__10_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr___closed__8_value),LEAN_SCALAR_PTR_LITERAL(250, 44, 198, 216, 184, 195, 199, 178)}};
static const lean_ctor_object lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr___closed__10_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr___closed__9_value),LEAN_SCALAR_PTR_LITERAL(117, 151, 161, 190, 111, 237, 188, 218)}};
static const lean_object* lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr___closed__10_value;
static lean_once_cell_t lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr___closed__11_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr___closed__11;
static const lean_string_object lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "true"};
static const lean_object* lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr___closed__12_value;
static const lean_ctor_object lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr___closed__13_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr___closed__8_value),LEAN_SCALAR_PTR_LITERAL(250, 44, 198, 216, 184, 195, 199, 178)}};
static const lean_ctor_object lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr___closed__13_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr___closed__12_value),LEAN_SCALAR_PTR_LITERAL(22, 245, 194, 28, 184, 9, 113, 128)}};
static const lean_object* lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr___closed__13_value;
static lean_once_cell_t lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr___closed__14_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr___closed__14;
static const lean_string_object lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "none"};
static const lean_object* lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr___closed__15 = (const lean_object*)&lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr___closed__15_value;
static const lean_ctor_object lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr___closed__16_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr___closed__16_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr___closed__16_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr___closed__1_value),LEAN_SCALAR_PTR_LITERAL(185, 145, 56, 44, 129, 38, 15, 45)}};
static const lean_ctor_object lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr___closed__16_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr___closed__15_value),LEAN_SCALAR_PTR_LITERAL(115, 123, 121, 187, 105, 61, 172, 219)}};
static const lean_object* lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr___closed__16 = (const lean_object*)&lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr___closed__16_value;
static lean_once_cell_t lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr___closed__17_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr___closed__17;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr(lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_instToExprSourceInfo__mathlib___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_instToExprSourceInfo__mathlib___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_instToExprSourceInfo__mathlib___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_instToExprSourceInfo__mathlib___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_instToExprSourceInfo__mathlib___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_instToExprSourceInfo__mathlib___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr___closed__1_value),LEAN_SCALAR_PTR_LITERAL(185, 145, 56, 44, 129, 38, 15, 45)}};
static const lean_object* lp_mathlib_Mathlib_instToExprSourceInfo__mathlib___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_instToExprSourceInfo__mathlib___closed__1_value;
static lean_once_cell_t lp_mathlib_Mathlib_instToExprSourceInfo__mathlib___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_instToExprSourceInfo__mathlib___closed__2;
static lean_once_cell_t lp_mathlib_Mathlib_instToExprSourceInfo__mathlib___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_instToExprSourceInfo__mathlib___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_instToExprSourceInfo__mathlib;
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00__private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprSyntax__mathlib_toExpr_spec__1_spec__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00__private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprSyntax__mathlib_toExpr_spec__1_spec__1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprSyntax__mathlib_toExpr_spec__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Syntax"};
static const lean_object* lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprSyntax__mathlib_toExpr_spec__1___closed__0 = (const lean_object*)&lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprSyntax__mathlib_toExpr_spec__1___closed__0_value;
static const lean_string_object lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprSyntax__mathlib_toExpr_spec__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "Preresolved"};
static const lean_object* lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprSyntax__mathlib_toExpr_spec__1___closed__1 = (const lean_object*)&lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprSyntax__mathlib_toExpr_spec__1___closed__1_value;
static const lean_string_object lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprSyntax__mathlib_toExpr_spec__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "namespace"};
static const lean_object* lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprSyntax__mathlib_toExpr_spec__1___closed__2 = (const lean_object*)&lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprSyntax__mathlib_toExpr_spec__1___closed__2_value;
static const lean_ctor_object lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprSyntax__mathlib_toExpr_spec__1___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprSyntax__mathlib_toExpr_spec__1___closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprSyntax__mathlib_toExpr_spec__1___closed__3_value_aux_0),((lean_object*)&lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprSyntax__mathlib_toExpr_spec__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(45, 144, 98, 72, 115, 31, 20, 74)}};
static const lean_ctor_object lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprSyntax__mathlib_toExpr_spec__1___closed__3_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprSyntax__mathlib_toExpr_spec__1___closed__3_value_aux_1),((lean_object*)&lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprSyntax__mathlib_toExpr_spec__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(203, 115, 25, 42, 173, 164, 230, 137)}};
static const lean_ctor_object lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprSyntax__mathlib_toExpr_spec__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprSyntax__mathlib_toExpr_spec__1___closed__3_value_aux_2),((lean_object*)&lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprSyntax__mathlib_toExpr_spec__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(141, 91, 234, 5, 195, 77, 204, 210)}};
static const lean_object* lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprSyntax__mathlib_toExpr_spec__1___closed__3 = (const lean_object*)&lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprSyntax__mathlib_toExpr_spec__1___closed__3_value;
static lean_once_cell_t lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprSyntax__mathlib_toExpr_spec__1___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprSyntax__mathlib_toExpr_spec__1___closed__4;
static const lean_string_object lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprSyntax__mathlib_toExpr_spec__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "decl"};
static const lean_object* lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprSyntax__mathlib_toExpr_spec__1___closed__5 = (const lean_object*)&lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprSyntax__mathlib_toExpr_spec__1___closed__5_value;
static const lean_ctor_object lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprSyntax__mathlib_toExpr_spec__1___closed__6_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprSyntax__mathlib_toExpr_spec__1___closed__6_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprSyntax__mathlib_toExpr_spec__1___closed__6_value_aux_0),((lean_object*)&lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprSyntax__mathlib_toExpr_spec__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(45, 144, 98, 72, 115, 31, 20, 74)}};
static const lean_ctor_object lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprSyntax__mathlib_toExpr_spec__1___closed__6_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprSyntax__mathlib_toExpr_spec__1___closed__6_value_aux_1),((lean_object*)&lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprSyntax__mathlib_toExpr_spec__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(203, 115, 25, 42, 173, 164, 230, 137)}};
static const lean_ctor_object lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprSyntax__mathlib_toExpr_spec__1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprSyntax__mathlib_toExpr_spec__1___closed__6_value_aux_2),((lean_object*)&lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprSyntax__mathlib_toExpr_spec__1___closed__5_value),LEAN_SCALAR_PTR_LITERAL(10, 43, 252, 229, 158, 70, 246, 135)}};
static const lean_object* lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprSyntax__mathlib_toExpr_spec__1___closed__6 = (const lean_object*)&lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprSyntax__mathlib_toExpr_spec__1___closed__6_value;
static lean_once_cell_t lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprSyntax__mathlib_toExpr_spec__1___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprSyntax__mathlib_toExpr_spec__1___closed__7;
static const lean_ctor_object lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprSyntax__mathlib_toExpr_spec__1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_instToExprRaw__mathlib_toExpr___closed__0_value),LEAN_SCALAR_PTR_LITERAL(6, 130, 56, 8, 41, 104, 134, 43)}};
static const lean_object* lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprSyntax__mathlib_toExpr_spec__1___closed__8 = (const lean_object*)&lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprSyntax__mathlib_toExpr_spec__1___closed__8_value;
static lean_once_cell_t lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprSyntax__mathlib_toExpr_spec__1___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprSyntax__mathlib_toExpr_spec__1___closed__9;
static const lean_string_object lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprSyntax__mathlib_toExpr_spec__1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "List"};
static const lean_object* lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprSyntax__mathlib_toExpr_spec__1___closed__10 = (const lean_object*)&lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprSyntax__mathlib_toExpr_spec__1___closed__10_value;
static const lean_string_object lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprSyntax__mathlib_toExpr_spec__1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "nil"};
static const lean_object* lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprSyntax__mathlib_toExpr_spec__1___closed__11 = (const lean_object*)&lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprSyntax__mathlib_toExpr_spec__1___closed__11_value;
static const lean_ctor_object lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprSyntax__mathlib_toExpr_spec__1___closed__12_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprSyntax__mathlib_toExpr_spec__1___closed__10_value),LEAN_SCALAR_PTR_LITERAL(245, 188, 225, 225, 165, 5, 251, 132)}};
static const lean_ctor_object lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprSyntax__mathlib_toExpr_spec__1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprSyntax__mathlib_toExpr_spec__1___closed__12_value_aux_0),((lean_object*)&lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprSyntax__mathlib_toExpr_spec__1___closed__11_value),LEAN_SCALAR_PTR_LITERAL(90, 150, 134, 113, 145, 38, 173, 251)}};
static const lean_object* lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprSyntax__mathlib_toExpr_spec__1___closed__12 = (const lean_object*)&lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprSyntax__mathlib_toExpr_spec__1___closed__12_value;
static const lean_ctor_object lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprSyntax__mathlib_toExpr_spec__1___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprSyntax__mathlib_toExpr_spec__1___closed__13 = (const lean_object*)&lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprSyntax__mathlib_toExpr_spec__1___closed__13_value;
static lean_once_cell_t lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprSyntax__mathlib_toExpr_spec__1___closed__14_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprSyntax__mathlib_toExpr_spec__1___closed__14;
static lean_once_cell_t lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprSyntax__mathlib_toExpr_spec__1___closed__15_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprSyntax__mathlib_toExpr_spec__1___closed__15;
static const lean_string_object lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprSyntax__mathlib_toExpr_spec__1___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "cons"};
static const lean_object* lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprSyntax__mathlib_toExpr_spec__1___closed__16 = (const lean_object*)&lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprSyntax__mathlib_toExpr_spec__1___closed__16_value;
static const lean_ctor_object lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprSyntax__mathlib_toExpr_spec__1___closed__17_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprSyntax__mathlib_toExpr_spec__1___closed__10_value),LEAN_SCALAR_PTR_LITERAL(245, 188, 225, 225, 165, 5, 251, 132)}};
static const lean_ctor_object lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprSyntax__mathlib_toExpr_spec__1___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprSyntax__mathlib_toExpr_spec__1___closed__17_value_aux_0),((lean_object*)&lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprSyntax__mathlib_toExpr_spec__1___closed__16_value),LEAN_SCALAR_PTR_LITERAL(98, 170, 59, 223, 79, 132, 139, 119)}};
static const lean_object* lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprSyntax__mathlib_toExpr_spec__1___closed__17 = (const lean_object*)&lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprSyntax__mathlib_toExpr_spec__1___closed__17_value;
static lean_once_cell_t lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprSyntax__mathlib_toExpr_spec__1___closed__18_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprSyntax__mathlib_toExpr_spec__1___closed__18;
static lean_once_cell_t lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprSyntax__mathlib_toExpr_spec__1___closed__19_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprSyntax__mathlib_toExpr_spec__1___closed__19;
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprSyntax__mathlib_toExpr_spec__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprSyntax__mathlib_toExpr_spec__1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_instToExprSyntax__mathlib_toExpr___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "missing"};
static const lean_object* lp_mathlib_Mathlib_instToExprSyntax__mathlib_toExpr___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_instToExprSyntax__mathlib_toExpr___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_instToExprSyntax__mathlib_toExpr___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_instToExprSyntax__mathlib_toExpr___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_instToExprSyntax__mathlib_toExpr___closed__1_value_aux_0),((lean_object*)&lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprSyntax__mathlib_toExpr_spec__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(45, 144, 98, 72, 115, 31, 20, 74)}};
static const lean_ctor_object lp_mathlib_Mathlib_instToExprSyntax__mathlib_toExpr___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_instToExprSyntax__mathlib_toExpr___closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_instToExprSyntax__mathlib_toExpr___closed__0_value),LEAN_SCALAR_PTR_LITERAL(146, 14, 86, 101, 181, 214, 128, 19)}};
static const lean_object* lp_mathlib_Mathlib_instToExprSyntax__mathlib_toExpr___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_instToExprSyntax__mathlib_toExpr___closed__1_value;
static lean_once_cell_t lp_mathlib_Mathlib_instToExprSyntax__mathlib_toExpr___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_instToExprSyntax__mathlib_toExpr___closed__2;
static const lean_ctor_object lp_mathlib_Mathlib_instToExprSyntax__mathlib_toExpr___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_instToExprSyntax__mathlib_toExpr___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_instToExprSyntax__mathlib_toExpr___closed__3_value_aux_0),((lean_object*)&lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprSyntax__mathlib_toExpr_spec__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(45, 144, 98, 72, 115, 31, 20, 74)}};
static const lean_object* lp_mathlib_Mathlib_instToExprSyntax__mathlib_toExpr___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_instToExprSyntax__mathlib_toExpr___closed__3_value;
static lean_once_cell_t lp_mathlib_Mathlib_instToExprSyntax__mathlib_toExpr___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_instToExprSyntax__mathlib_toExpr___closed__4;
static const lean_string_object lp_mathlib_Mathlib_instToExprSyntax__mathlib_toExpr___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "node"};
static const lean_object* lp_mathlib_Mathlib_instToExprSyntax__mathlib_toExpr___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_instToExprSyntax__mathlib_toExpr___closed__5_value;
static const lean_ctor_object lp_mathlib_Mathlib_instToExprSyntax__mathlib_toExpr___closed__6_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_instToExprSyntax__mathlib_toExpr___closed__6_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_instToExprSyntax__mathlib_toExpr___closed__6_value_aux_0),((lean_object*)&lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprSyntax__mathlib_toExpr_spec__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(45, 144, 98, 72, 115, 31, 20, 74)}};
static const lean_ctor_object lp_mathlib_Mathlib_instToExprSyntax__mathlib_toExpr___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_instToExprSyntax__mathlib_toExpr___closed__6_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_instToExprSyntax__mathlib_toExpr___closed__5_value),LEAN_SCALAR_PTR_LITERAL(238, 246, 230, 219, 171, 12, 18, 4)}};
static const lean_object* lp_mathlib_Mathlib_instToExprSyntax__mathlib_toExpr___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_instToExprSyntax__mathlib_toExpr___closed__6_value;
static lean_once_cell_t lp_mathlib_Mathlib_instToExprSyntax__mathlib_toExpr___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_instToExprSyntax__mathlib_toExpr___closed__7;
static const lean_string_object lp_mathlib_Mathlib_instToExprSyntax__mathlib_toExpr___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "toArray"};
static const lean_object* lp_mathlib_Mathlib_instToExprSyntax__mathlib_toExpr___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_instToExprSyntax__mathlib_toExpr___closed__8_value;
static const lean_ctor_object lp_mathlib_Mathlib_instToExprSyntax__mathlib_toExpr___closed__9_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprSyntax__mathlib_toExpr_spec__1___closed__10_value),LEAN_SCALAR_PTR_LITERAL(245, 188, 225, 225, 165, 5, 251, 132)}};
static const lean_ctor_object lp_mathlib_Mathlib_instToExprSyntax__mathlib_toExpr___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_instToExprSyntax__mathlib_toExpr___closed__9_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_instToExprSyntax__mathlib_toExpr___closed__8_value),LEAN_SCALAR_PTR_LITERAL(225, 54, 189, 64, 249, 49, 198, 116)}};
static const lean_object* lp_mathlib_Mathlib_instToExprSyntax__mathlib_toExpr___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_instToExprSyntax__mathlib_toExpr___closed__9_value;
static lean_once_cell_t lp_mathlib_Mathlib_instToExprSyntax__mathlib_toExpr___closed__10_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_instToExprSyntax__mathlib_toExpr___closed__10;
static lean_once_cell_t lp_mathlib_Mathlib_instToExprSyntax__mathlib_toExpr___closed__11_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_instToExprSyntax__mathlib_toExpr___closed__11;
static lean_once_cell_t lp_mathlib_Mathlib_instToExprSyntax__mathlib_toExpr___closed__12_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_instToExprSyntax__mathlib_toExpr___closed__12;
static const lean_string_object lp_mathlib_Mathlib_instToExprSyntax__mathlib_toExpr___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "atom"};
static const lean_object* lp_mathlib_Mathlib_instToExprSyntax__mathlib_toExpr___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_instToExprSyntax__mathlib_toExpr___closed__13_value;
static const lean_ctor_object lp_mathlib_Mathlib_instToExprSyntax__mathlib_toExpr___closed__14_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_instToExprSyntax__mathlib_toExpr___closed__14_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_instToExprSyntax__mathlib_toExpr___closed__14_value_aux_0),((lean_object*)&lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprSyntax__mathlib_toExpr_spec__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(45, 144, 98, 72, 115, 31, 20, 74)}};
static const lean_ctor_object lp_mathlib_Mathlib_instToExprSyntax__mathlib_toExpr___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_instToExprSyntax__mathlib_toExpr___closed__14_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_instToExprSyntax__mathlib_toExpr___closed__13_value),LEAN_SCALAR_PTR_LITERAL(105, 146, 243, 197, 13, 246, 171, 67)}};
static const lean_object* lp_mathlib_Mathlib_instToExprSyntax__mathlib_toExpr___closed__14 = (const lean_object*)&lp_mathlib_Mathlib_instToExprSyntax__mathlib_toExpr___closed__14_value;
static lean_once_cell_t lp_mathlib_Mathlib_instToExprSyntax__mathlib_toExpr___closed__15_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_instToExprSyntax__mathlib_toExpr___closed__15;
static const lean_string_object lp_mathlib_Mathlib_instToExprSyntax__mathlib_toExpr___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ident"};
static const lean_object* lp_mathlib_Mathlib_instToExprSyntax__mathlib_toExpr___closed__16 = (const lean_object*)&lp_mathlib_Mathlib_instToExprSyntax__mathlib_toExpr___closed__16_value;
static const lean_ctor_object lp_mathlib_Mathlib_instToExprSyntax__mathlib_toExpr___closed__17_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_instToExprSyntax__mathlib_toExpr___closed__17_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_instToExprSyntax__mathlib_toExpr___closed__17_value_aux_0),((lean_object*)&lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprSyntax__mathlib_toExpr_spec__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(45, 144, 98, 72, 115, 31, 20, 74)}};
static const lean_ctor_object lp_mathlib_Mathlib_instToExprSyntax__mathlib_toExpr___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_instToExprSyntax__mathlib_toExpr___closed__17_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_instToExprSyntax__mathlib_toExpr___closed__16_value),LEAN_SCALAR_PTR_LITERAL(218, 53, 19, 110, 198, 32, 158, 17)}};
static const lean_object* lp_mathlib_Mathlib_instToExprSyntax__mathlib_toExpr___closed__17 = (const lean_object*)&lp_mathlib_Mathlib_instToExprSyntax__mathlib_toExpr___closed__17_value;
static lean_once_cell_t lp_mathlib_Mathlib_instToExprSyntax__mathlib_toExpr___closed__18_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_instToExprSyntax__mathlib_toExpr___closed__18;
static const lean_ctor_object lp_mathlib_Mathlib_instToExprSyntax__mathlib_toExpr___closed__19_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_instToExprSyntax__mathlib_toExpr___closed__19_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_instToExprSyntax__mathlib_toExpr___closed__19_value_aux_0),((lean_object*)&lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprSyntax__mathlib_toExpr_spec__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(45, 144, 98, 72, 115, 31, 20, 74)}};
static const lean_ctor_object lp_mathlib_Mathlib_instToExprSyntax__mathlib_toExpr___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_instToExprSyntax__mathlib_toExpr___closed__19_value_aux_1),((lean_object*)&lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprSyntax__mathlib_toExpr_spec__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(203, 115, 25, 42, 173, 164, 230, 137)}};
static const lean_object* lp_mathlib_Mathlib_instToExprSyntax__mathlib_toExpr___closed__19 = (const lean_object*)&lp_mathlib_Mathlib_instToExprSyntax__mathlib_toExpr___closed__19_value;
static lean_once_cell_t lp_mathlib_Mathlib_instToExprSyntax__mathlib_toExpr___closed__20_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_instToExprSyntax__mathlib_toExpr___closed__20;
static lean_once_cell_t lp_mathlib_Mathlib_instToExprSyntax__mathlib_toExpr___closed__21_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_instToExprSyntax__mathlib_toExpr___closed__21;
static lean_once_cell_t lp_mathlib_Mathlib_instToExprSyntax__mathlib_toExpr___closed__22_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_instToExprSyntax__mathlib_toExpr___closed__22;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_instToExprSyntax__mathlib_toExpr(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprSyntax__mathlib_toExpr_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprSyntax__mathlib_toExpr_spec__0___boxed(lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_instToExprSyntax__mathlib___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_instToExprSyntax__mathlib_toExpr, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_instToExprSyntax__mathlib___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_instToExprSyntax__mathlib___closed__0_value;
static lean_once_cell_t lp_mathlib_Mathlib_instToExprSyntax__mathlib___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_instToExprSyntax__mathlib___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_instToExprSyntax__mathlib;
static const lean_string_object lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "KVMap"};
static const lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__0 = (const lean_object*)&lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__0_value;
static const lean_string_object lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "setString"};
static const lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__1 = (const lean_object*)&lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__1_value;
static const lean_ctor_object lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__2_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__2_value_aux_0),((lean_object*)&lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(151, 95, 92, 183, 154, 214, 93, 213)}};
static const lean_ctor_object lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__2_value_aux_1),((lean_object*)&lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__1_value),LEAN_SCALAR_PTR_LITERAL(189, 35, 57, 164, 139, 58, 100, 137)}};
static const lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__2 = (const lean_object*)&lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__2_value;
static lean_once_cell_t lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__3;
static const lean_string_object lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "setBool"};
static const lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__4 = (const lean_object*)&lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__4_value;
static const lean_ctor_object lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__5_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__5_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__5_value_aux_0),((lean_object*)&lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(151, 95, 92, 183, 154, 214, 93, 213)}};
static const lean_ctor_object lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__5_value_aux_1),((lean_object*)&lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__4_value),LEAN_SCALAR_PTR_LITERAL(16, 228, 239, 150, 219, 122, 206, 148)}};
static const lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__5 = (const lean_object*)&lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__5_value;
static lean_once_cell_t lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__6;
static const lean_string_object lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "setName"};
static const lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__7 = (const lean_object*)&lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__7_value;
static const lean_ctor_object lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__8_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__8_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__8_value_aux_0),((lean_object*)&lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(151, 95, 92, 183, 154, 214, 93, 213)}};
static const lean_ctor_object lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__8_value_aux_1),((lean_object*)&lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__7_value),LEAN_SCALAR_PTR_LITERAL(89, 147, 223, 143, 38, 139, 139, 74)}};
static const lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__8 = (const lean_object*)&lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__8_value;
static lean_once_cell_t lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__9;
static const lean_string_object lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "setNat"};
static const lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__10 = (const lean_object*)&lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__10_value;
static const lean_ctor_object lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__11_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__11_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__11_value_aux_0),((lean_object*)&lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(151, 95, 92, 183, 154, 214, 93, 213)}};
static const lean_ctor_object lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__11_value_aux_1),((lean_object*)&lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__10_value),LEAN_SCALAR_PTR_LITERAL(158, 235, 43, 39, 180, 27, 224, 41)}};
static const lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__11 = (const lean_object*)&lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__11_value;
static lean_once_cell_t lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__12_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__12;
static const lean_string_object lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "setInt"};
static const lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__13 = (const lean_object*)&lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__13_value;
static const lean_ctor_object lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__14_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__14_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__14_value_aux_0),((lean_object*)&lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(151, 95, 92, 183, 154, 214, 93, 213)}};
static const lean_ctor_object lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__14_value_aux_1),((lean_object*)&lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__13_value),LEAN_SCALAR_PTR_LITERAL(59, 229, 98, 11, 115, 199, 151, 183)}};
static const lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__14 = (const lean_object*)&lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__14_value;
static lean_once_cell_t lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__15_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__15;
static lean_once_cell_t lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__16_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__16;
static const lean_string_object lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Neg"};
static const lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__17 = (const lean_object*)&lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__17_value;
static const lean_string_object lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "neg"};
static const lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__18 = (const lean_object*)&lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__18_value;
static const lean_ctor_object lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__19_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__17_value),LEAN_SCALAR_PTR_LITERAL(94, 4, 109, 108, 64, 81, 153, 133)}};
static const lean_ctor_object lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__19_value_aux_0),((lean_object*)&lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__18_value),LEAN_SCALAR_PTR_LITERAL(105, 26, 70, 221, 245, 238, 127, 238)}};
static const lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__19 = (const lean_object*)&lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__19_value;
static lean_once_cell_t lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__20_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__20;
static lean_once_cell_t lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__21_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__21;
static lean_once_cell_t lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__22_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__22;
static const lean_string_object lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Int"};
static const lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__23 = (const lean_object*)&lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__23_value;
static const lean_ctor_object lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__23_value),LEAN_SCALAR_PTR_LITERAL(61, 25, 98, 154, 117, 127, 69, 97)}};
static const lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__24 = (const lean_object*)&lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__24_value;
static lean_once_cell_t lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__25_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__25;
static const lean_string_object lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__26_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "instNegInt"};
static const lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__26 = (const lean_object*)&lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__26_value;
static const lean_ctor_object lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__27_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__23_value),LEAN_SCALAR_PTR_LITERAL(61, 25, 98, 154, 117, 127, 69, 97)}};
static const lean_ctor_object lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__27_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__27_value_aux_0),((lean_object*)&lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__26_value),LEAN_SCALAR_PTR_LITERAL(217, 109, 233, 1, 211, 122, 77, 88)}};
static const lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__27 = (const lean_object*)&lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__27_value;
static lean_once_cell_t lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__28_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__28;
static const lean_string_object lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__29_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "setSyntax"};
static const lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__29 = (const lean_object*)&lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__29_value;
static const lean_ctor_object lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__30_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__30_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__30_value_aux_0),((lean_object*)&lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(151, 95, 92, 183, 154, 214, 93, 213)}};
static const lean_ctor_object lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__30_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__30_value_aux_1),((lean_object*)&lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__29_value),LEAN_SCALAR_PTR_LITERAL(232, 223, 72, 148, 202, 145, 61, 248)}};
static const lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__30 = (const lean_object*)&lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__30_value;
static lean_once_cell_t lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__31_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__31;
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___boxed(lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "MData"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData___closed__0_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "empty"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData___closed__1_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData___closed__2_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData___closed__2_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData___closed__0_value),LEAN_SCALAR_PTR_LITERAL(146, 59, 178, 55, 17, 203, 226, 154)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData___closed__2_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData___closed__1_value),LEAN_SCALAR_PTR_LITERAL(202, 189, 63, 250, 0, 123, 75, 60)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData___closed__2_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData___closed__3;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_instToExprMData__mathlib___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_instToExprMData__mathlib___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_instToExprMData__mathlib___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_instToExprMData__mathlib___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_instToExprMData__mathlib___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_instToExprMData__mathlib___closed__1_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData___closed__0_value),LEAN_SCALAR_PTR_LITERAL(146, 59, 178, 55, 17, 203, 226, 154)}};
static const lean_object* lp_mathlib_Mathlib_instToExprMData__mathlib___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_instToExprMData__mathlib___closed__1_value;
static lean_once_cell_t lp_mathlib_Mathlib_instToExprMData__mathlib___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_instToExprMData__mathlib___closed__2;
static lean_once_cell_t lp_mathlib_Mathlib_instToExprMData__mathlib___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_instToExprMData__mathlib___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_instToExprMData__mathlib;
static const lean_string_object lp_mathlib_Mathlib_instToExprMVarId__mathlib_toExpr___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "MVarId"};
static const lean_object* lp_mathlib_Mathlib_instToExprMVarId__mathlib_toExpr___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_instToExprMVarId__mathlib_toExpr___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_instToExprMVarId__mathlib_toExpr___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_instToExprMVarId__mathlib_toExpr___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_instToExprMVarId__mathlib_toExpr___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_instToExprMVarId__mathlib_toExpr___closed__0_value),LEAN_SCALAR_PTR_LITERAL(177, 186, 234, 138, 172, 166, 87, 74)}};
static const lean_ctor_object lp_mathlib_Mathlib_instToExprMVarId__mathlib_toExpr___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_instToExprMVarId__mathlib_toExpr___closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_instToExprRaw__mathlib_toExpr___closed__3_value),LEAN_SCALAR_PTR_LITERAL(93, 44, 60, 136, 72, 250, 230, 141)}};
static const lean_object* lp_mathlib_Mathlib_instToExprMVarId__mathlib_toExpr___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_instToExprMVarId__mathlib_toExpr___closed__1_value;
static lean_once_cell_t lp_mathlib_Mathlib_instToExprMVarId__mathlib_toExpr___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_instToExprMVarId__mathlib_toExpr___closed__2;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_instToExprMVarId__mathlib_toExpr(lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_instToExprMVarId__mathlib___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_instToExprMVarId__mathlib_toExpr, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_instToExprMVarId__mathlib___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_instToExprMVarId__mathlib___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_instToExprMVarId__mathlib___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_instToExprMVarId__mathlib___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_instToExprMVarId__mathlib___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_instToExprMVarId__mathlib_toExpr___closed__0_value),LEAN_SCALAR_PTR_LITERAL(177, 186, 234, 138, 172, 166, 87, 74)}};
static const lean_object* lp_mathlib_Mathlib_instToExprMVarId__mathlib___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_instToExprMVarId__mathlib___closed__1_value;
static lean_once_cell_t lp_mathlib_Mathlib_instToExprMVarId__mathlib___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_instToExprMVarId__mathlib___closed__2;
static lean_once_cell_t lp_mathlib_Mathlib_instToExprMVarId__mathlib___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_instToExprMVarId__mathlib___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_instToExprMVarId__mathlib;
static const lean_string_object lp_mathlib_Mathlib_instToExprLevelMVarId__mathlib_toExpr___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "LevelMVarId"};
static const lean_object* lp_mathlib_Mathlib_instToExprLevelMVarId__mathlib_toExpr___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_instToExprLevelMVarId__mathlib_toExpr___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_instToExprLevelMVarId__mathlib_toExpr___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_instToExprLevelMVarId__mathlib_toExpr___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_instToExprLevelMVarId__mathlib_toExpr___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_instToExprLevelMVarId__mathlib_toExpr___closed__0_value),LEAN_SCALAR_PTR_LITERAL(89, 60, 85, 89, 175, 240, 129, 147)}};
static const lean_ctor_object lp_mathlib_Mathlib_instToExprLevelMVarId__mathlib_toExpr___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_instToExprLevelMVarId__mathlib_toExpr___closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_instToExprRaw__mathlib_toExpr___closed__3_value),LEAN_SCALAR_PTR_LITERAL(213, 157, 226, 48, 182, 72, 20, 234)}};
static const lean_object* lp_mathlib_Mathlib_instToExprLevelMVarId__mathlib_toExpr___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_instToExprLevelMVarId__mathlib_toExpr___closed__1_value;
static lean_once_cell_t lp_mathlib_Mathlib_instToExprLevelMVarId__mathlib_toExpr___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_instToExprLevelMVarId__mathlib_toExpr___closed__2;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_instToExprLevelMVarId__mathlib_toExpr(lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_instToExprLevelMVarId__mathlib___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_instToExprLevelMVarId__mathlib_toExpr, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_instToExprLevelMVarId__mathlib___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_instToExprLevelMVarId__mathlib___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_instToExprLevelMVarId__mathlib___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_instToExprLevelMVarId__mathlib___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_instToExprLevelMVarId__mathlib___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_instToExprLevelMVarId__mathlib_toExpr___closed__0_value),LEAN_SCALAR_PTR_LITERAL(89, 60, 85, 89, 175, 240, 129, 147)}};
static const lean_object* lp_mathlib_Mathlib_instToExprLevelMVarId__mathlib___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_instToExprLevelMVarId__mathlib___closed__1_value;
static lean_once_cell_t lp_mathlib_Mathlib_instToExprLevelMVarId__mathlib___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_instToExprLevelMVarId__mathlib___closed__2;
static lean_once_cell_t lp_mathlib_Mathlib_instToExprLevelMVarId__mathlib___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_instToExprLevelMVarId__mathlib___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_instToExprLevelMVarId__mathlib;
static const lean_string_object lp_mathlib_Mathlib_instToExprLevel__mathlib_toExpr___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "Level"};
static const lean_object* lp_mathlib_Mathlib_instToExprLevel__mathlib_toExpr___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_instToExprLevel__mathlib_toExpr___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_instToExprLevel__mathlib_toExpr___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "zero"};
static const lean_object* lp_mathlib_Mathlib_instToExprLevel__mathlib_toExpr___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_instToExprLevel__mathlib_toExpr___closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_instToExprLevel__mathlib_toExpr___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_instToExprLevel__mathlib_toExpr___closed__2_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_instToExprLevel__mathlib_toExpr___closed__2_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_instToExprLevel__mathlib_toExpr___closed__0_value),LEAN_SCALAR_PTR_LITERAL(165, 140, 25, 163, 179, 32, 245, 95)}};
static const lean_ctor_object lp_mathlib_Mathlib_instToExprLevel__mathlib_toExpr___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_instToExprLevel__mathlib_toExpr___closed__2_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_instToExprLevel__mathlib_toExpr___closed__1_value),LEAN_SCALAR_PTR_LITERAL(13, 6, 49, 141, 220, 30, 84, 149)}};
static const lean_object* lp_mathlib_Mathlib_instToExprLevel__mathlib_toExpr___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_instToExprLevel__mathlib_toExpr___closed__2_value;
static lean_once_cell_t lp_mathlib_Mathlib_instToExprLevel__mathlib_toExpr___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_instToExprLevel__mathlib_toExpr___closed__3;
static const lean_string_object lp_mathlib_Mathlib_instToExprLevel__mathlib_toExpr___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "succ"};
static const lean_object* lp_mathlib_Mathlib_instToExprLevel__mathlib_toExpr___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_instToExprLevel__mathlib_toExpr___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_instToExprLevel__mathlib_toExpr___closed__5_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_instToExprLevel__mathlib_toExpr___closed__5_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_instToExprLevel__mathlib_toExpr___closed__5_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_instToExprLevel__mathlib_toExpr___closed__0_value),LEAN_SCALAR_PTR_LITERAL(165, 140, 25, 163, 179, 32, 245, 95)}};
static const lean_ctor_object lp_mathlib_Mathlib_instToExprLevel__mathlib_toExpr___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_instToExprLevel__mathlib_toExpr___closed__5_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_instToExprLevel__mathlib_toExpr___closed__4_value),LEAN_SCALAR_PTR_LITERAL(227, 93, 133, 102, 36, 205, 79, 205)}};
static const lean_object* lp_mathlib_Mathlib_instToExprLevel__mathlib_toExpr___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_instToExprLevel__mathlib_toExpr___closed__5_value;
static lean_once_cell_t lp_mathlib_Mathlib_instToExprLevel__mathlib_toExpr___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_instToExprLevel__mathlib_toExpr___closed__6;
static const lean_string_object lp_mathlib_Mathlib_instToExprLevel__mathlib_toExpr___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "max"};
static const lean_object* lp_mathlib_Mathlib_instToExprLevel__mathlib_toExpr___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_instToExprLevel__mathlib_toExpr___closed__7_value;
static const lean_ctor_object lp_mathlib_Mathlib_instToExprLevel__mathlib_toExpr___closed__8_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_instToExprLevel__mathlib_toExpr___closed__8_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_instToExprLevel__mathlib_toExpr___closed__8_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_instToExprLevel__mathlib_toExpr___closed__0_value),LEAN_SCALAR_PTR_LITERAL(165, 140, 25, 163, 179, 32, 245, 95)}};
static const lean_ctor_object lp_mathlib_Mathlib_instToExprLevel__mathlib_toExpr___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_instToExprLevel__mathlib_toExpr___closed__8_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_instToExprLevel__mathlib_toExpr___closed__7_value),LEAN_SCALAR_PTR_LITERAL(163, 196, 232, 122, 251, 166, 170, 227)}};
static const lean_object* lp_mathlib_Mathlib_instToExprLevel__mathlib_toExpr___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_instToExprLevel__mathlib_toExpr___closed__8_value;
static lean_once_cell_t lp_mathlib_Mathlib_instToExprLevel__mathlib_toExpr___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_instToExprLevel__mathlib_toExpr___closed__9;
static const lean_string_object lp_mathlib_Mathlib_instToExprLevel__mathlib_toExpr___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "imax"};
static const lean_object* lp_mathlib_Mathlib_instToExprLevel__mathlib_toExpr___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_instToExprLevel__mathlib_toExpr___closed__10_value;
static const lean_ctor_object lp_mathlib_Mathlib_instToExprLevel__mathlib_toExpr___closed__11_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_instToExprLevel__mathlib_toExpr___closed__11_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_instToExprLevel__mathlib_toExpr___closed__11_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_instToExprLevel__mathlib_toExpr___closed__0_value),LEAN_SCALAR_PTR_LITERAL(165, 140, 25, 163, 179, 32, 245, 95)}};
static const lean_ctor_object lp_mathlib_Mathlib_instToExprLevel__mathlib_toExpr___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_instToExprLevel__mathlib_toExpr___closed__11_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_instToExprLevel__mathlib_toExpr___closed__10_value),LEAN_SCALAR_PTR_LITERAL(13, 164, 87, 20, 224, 129, 213, 91)}};
static const lean_object* lp_mathlib_Mathlib_instToExprLevel__mathlib_toExpr___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_instToExprLevel__mathlib_toExpr___closed__11_value;
static lean_once_cell_t lp_mathlib_Mathlib_instToExprLevel__mathlib_toExpr___closed__12_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_instToExprLevel__mathlib_toExpr___closed__12;
static const lean_string_object lp_mathlib_Mathlib_instToExprLevel__mathlib_toExpr___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "param"};
static const lean_object* lp_mathlib_Mathlib_instToExprLevel__mathlib_toExpr___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_instToExprLevel__mathlib_toExpr___closed__13_value;
static const lean_ctor_object lp_mathlib_Mathlib_instToExprLevel__mathlib_toExpr___closed__14_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_instToExprLevel__mathlib_toExpr___closed__14_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_instToExprLevel__mathlib_toExpr___closed__14_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_instToExprLevel__mathlib_toExpr___closed__0_value),LEAN_SCALAR_PTR_LITERAL(165, 140, 25, 163, 179, 32, 245, 95)}};
static const lean_ctor_object lp_mathlib_Mathlib_instToExprLevel__mathlib_toExpr___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_instToExprLevel__mathlib_toExpr___closed__14_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_instToExprLevel__mathlib_toExpr___closed__13_value),LEAN_SCALAR_PTR_LITERAL(196, 134, 94, 195, 247, 235, 245, 84)}};
static const lean_object* lp_mathlib_Mathlib_instToExprLevel__mathlib_toExpr___closed__14 = (const lean_object*)&lp_mathlib_Mathlib_instToExprLevel__mathlib_toExpr___closed__14_value;
static lean_once_cell_t lp_mathlib_Mathlib_instToExprLevel__mathlib_toExpr___closed__15_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_instToExprLevel__mathlib_toExpr___closed__15;
static const lean_string_object lp_mathlib_Mathlib_instToExprLevel__mathlib_toExpr___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "mvar"};
static const lean_object* lp_mathlib_Mathlib_instToExprLevel__mathlib_toExpr___closed__16 = (const lean_object*)&lp_mathlib_Mathlib_instToExprLevel__mathlib_toExpr___closed__16_value;
static const lean_ctor_object lp_mathlib_Mathlib_instToExprLevel__mathlib_toExpr___closed__17_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_instToExprLevel__mathlib_toExpr___closed__17_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_instToExprLevel__mathlib_toExpr___closed__17_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_instToExprLevel__mathlib_toExpr___closed__0_value),LEAN_SCALAR_PTR_LITERAL(165, 140, 25, 163, 179, 32, 245, 95)}};
static const lean_ctor_object lp_mathlib_Mathlib_instToExprLevel__mathlib_toExpr___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_instToExprLevel__mathlib_toExpr___closed__17_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_instToExprLevel__mathlib_toExpr___closed__16_value),LEAN_SCALAR_PTR_LITERAL(33, 188, 104, 40, 236, 34, 24, 77)}};
static const lean_object* lp_mathlib_Mathlib_instToExprLevel__mathlib_toExpr___closed__17 = (const lean_object*)&lp_mathlib_Mathlib_instToExprLevel__mathlib_toExpr___closed__17_value;
static lean_once_cell_t lp_mathlib_Mathlib_instToExprLevel__mathlib_toExpr___closed__18_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_instToExprLevel__mathlib_toExpr___closed__18;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_instToExprLevel__mathlib_toExpr(lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_instToExprLevel__mathlib___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_instToExprLevel__mathlib_toExpr, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_instToExprLevel__mathlib___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_instToExprLevel__mathlib___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_instToExprLevel__mathlib___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_instToExprLevel__mathlib___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_instToExprLevel__mathlib___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_instToExprLevel__mathlib_toExpr___closed__0_value),LEAN_SCALAR_PTR_LITERAL(165, 140, 25, 163, 179, 32, 245, 95)}};
static const lean_object* lp_mathlib_Mathlib_instToExprLevel__mathlib___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_instToExprLevel__mathlib___closed__1_value;
static lean_once_cell_t lp_mathlib_Mathlib_instToExprLevel__mathlib___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_instToExprLevel__mathlib___closed__2;
static lean_once_cell_t lp_mathlib_Mathlib_instToExprLevel__mathlib___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_instToExprLevel__mathlib___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_instToExprLevel__mathlib;
static const lean_string_object lp_mathlib_Mathlib_instToExprBinderInfo__mathlib_toExpr___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "BinderInfo"};
static const lean_object* lp_mathlib_Mathlib_instToExprBinderInfo__mathlib_toExpr___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_instToExprBinderInfo__mathlib_toExpr___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_instToExprBinderInfo__mathlib_toExpr___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "default"};
static const lean_object* lp_mathlib_Mathlib_instToExprBinderInfo__mathlib_toExpr___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_instToExprBinderInfo__mathlib_toExpr___closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_instToExprBinderInfo__mathlib_toExpr___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_instToExprBinderInfo__mathlib_toExpr___closed__2_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_instToExprBinderInfo__mathlib_toExpr___closed__2_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_instToExprBinderInfo__mathlib_toExpr___closed__0_value),LEAN_SCALAR_PTR_LITERAL(25, 16, 226, 245, 106, 161, 19, 213)}};
static const lean_ctor_object lp_mathlib_Mathlib_instToExprBinderInfo__mathlib_toExpr___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_instToExprBinderInfo__mathlib_toExpr___closed__2_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_instToExprBinderInfo__mathlib_toExpr___closed__1_value),LEAN_SCALAR_PTR_LITERAL(39, 15, 252, 127, 213, 76, 105, 203)}};
static const lean_object* lp_mathlib_Mathlib_instToExprBinderInfo__mathlib_toExpr___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_instToExprBinderInfo__mathlib_toExpr___closed__2_value;
static lean_once_cell_t lp_mathlib_Mathlib_instToExprBinderInfo__mathlib_toExpr___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_instToExprBinderInfo__mathlib_toExpr___closed__3;
static const lean_string_object lp_mathlib_Mathlib_instToExprBinderInfo__mathlib_toExpr___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "implicit"};
static const lean_object* lp_mathlib_Mathlib_instToExprBinderInfo__mathlib_toExpr___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_instToExprBinderInfo__mathlib_toExpr___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_instToExprBinderInfo__mathlib_toExpr___closed__5_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_instToExprBinderInfo__mathlib_toExpr___closed__5_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_instToExprBinderInfo__mathlib_toExpr___closed__5_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_instToExprBinderInfo__mathlib_toExpr___closed__0_value),LEAN_SCALAR_PTR_LITERAL(25, 16, 226, 245, 106, 161, 19, 213)}};
static const lean_ctor_object lp_mathlib_Mathlib_instToExprBinderInfo__mathlib_toExpr___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_instToExprBinderInfo__mathlib_toExpr___closed__5_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_instToExprBinderInfo__mathlib_toExpr___closed__4_value),LEAN_SCALAR_PTR_LITERAL(251, 202, 67, 228, 64, 219, 133, 236)}};
static const lean_object* lp_mathlib_Mathlib_instToExprBinderInfo__mathlib_toExpr___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_instToExprBinderInfo__mathlib_toExpr___closed__5_value;
static lean_once_cell_t lp_mathlib_Mathlib_instToExprBinderInfo__mathlib_toExpr___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_instToExprBinderInfo__mathlib_toExpr___closed__6;
static const lean_string_object lp_mathlib_Mathlib_instToExprBinderInfo__mathlib_toExpr___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "strictImplicit"};
static const lean_object* lp_mathlib_Mathlib_instToExprBinderInfo__mathlib_toExpr___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_instToExprBinderInfo__mathlib_toExpr___closed__7_value;
static const lean_ctor_object lp_mathlib_Mathlib_instToExprBinderInfo__mathlib_toExpr___closed__8_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_instToExprBinderInfo__mathlib_toExpr___closed__8_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_instToExprBinderInfo__mathlib_toExpr___closed__8_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_instToExprBinderInfo__mathlib_toExpr___closed__0_value),LEAN_SCALAR_PTR_LITERAL(25, 16, 226, 245, 106, 161, 19, 213)}};
static const lean_ctor_object lp_mathlib_Mathlib_instToExprBinderInfo__mathlib_toExpr___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_instToExprBinderInfo__mathlib_toExpr___closed__8_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_instToExprBinderInfo__mathlib_toExpr___closed__7_value),LEAN_SCALAR_PTR_LITERAL(210, 36, 185, 75, 137, 139, 69, 221)}};
static const lean_object* lp_mathlib_Mathlib_instToExprBinderInfo__mathlib_toExpr___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_instToExprBinderInfo__mathlib_toExpr___closed__8_value;
static lean_once_cell_t lp_mathlib_Mathlib_instToExprBinderInfo__mathlib_toExpr___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_instToExprBinderInfo__mathlib_toExpr___closed__9;
static const lean_string_object lp_mathlib_Mathlib_instToExprBinderInfo__mathlib_toExpr___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "instImplicit"};
static const lean_object* lp_mathlib_Mathlib_instToExprBinderInfo__mathlib_toExpr___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_instToExprBinderInfo__mathlib_toExpr___closed__10_value;
static const lean_ctor_object lp_mathlib_Mathlib_instToExprBinderInfo__mathlib_toExpr___closed__11_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_instToExprBinderInfo__mathlib_toExpr___closed__11_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_instToExprBinderInfo__mathlib_toExpr___closed__11_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_instToExprBinderInfo__mathlib_toExpr___closed__0_value),LEAN_SCALAR_PTR_LITERAL(25, 16, 226, 245, 106, 161, 19, 213)}};
static const lean_ctor_object lp_mathlib_Mathlib_instToExprBinderInfo__mathlib_toExpr___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_instToExprBinderInfo__mathlib_toExpr___closed__11_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_instToExprBinderInfo__mathlib_toExpr___closed__10_value),LEAN_SCALAR_PTR_LITERAL(121, 65, 31, 57, 146, 51, 125, 181)}};
static const lean_object* lp_mathlib_Mathlib_instToExprBinderInfo__mathlib_toExpr___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_instToExprBinderInfo__mathlib_toExpr___closed__11_value;
static lean_once_cell_t lp_mathlib_Mathlib_instToExprBinderInfo__mathlib_toExpr___closed__12_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_instToExprBinderInfo__mathlib_toExpr___closed__12;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_instToExprBinderInfo__mathlib_toExpr(uint8_t);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_instToExprBinderInfo__mathlib_toExpr___boxed(lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_instToExprBinderInfo__mathlib___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_instToExprBinderInfo__mathlib_toExpr___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_instToExprBinderInfo__mathlib___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_instToExprBinderInfo__mathlib___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_instToExprBinderInfo__mathlib___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_instToExprBinderInfo__mathlib___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_instToExprBinderInfo__mathlib___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_instToExprBinderInfo__mathlib_toExpr___closed__0_value),LEAN_SCALAR_PTR_LITERAL(25, 16, 226, 245, 106, 161, 19, 213)}};
static const lean_object* lp_mathlib_Mathlib_instToExprBinderInfo__mathlib___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_instToExprBinderInfo__mathlib___closed__1_value;
static lean_once_cell_t lp_mathlib_Mathlib_instToExprBinderInfo__mathlib___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_instToExprBinderInfo__mathlib___closed__2;
static lean_once_cell_t lp_mathlib_Mathlib_instToExprBinderInfo__mathlib___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_instToExprBinderInfo__mathlib___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_instToExprBinderInfo__mathlib;
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprExpr__mathlib_toExpr_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprExpr__mathlib_toExpr_spec__0___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Expr"};
static const lean_object* lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "bvar"};
static const lean_object* lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__2_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__2_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__0_value),LEAN_SCALAR_PTR_LITERAL(84, 208, 74, 211, 93, 83, 88, 82)}};
static const lean_ctor_object lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__2_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__1_value),LEAN_SCALAR_PTR_LITERAL(194, 116, 189, 113, 236, 234, 204, 95)}};
static const lean_object* lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__2_value;
static lean_once_cell_t lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__3;
static const lean_string_object lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "fvar"};
static const lean_object* lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__5_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__5_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__5_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__0_value),LEAN_SCALAR_PTR_LITERAL(84, 208, 74, 211, 93, 83, 88, 82)}};
static const lean_ctor_object lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__5_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__4_value),LEAN_SCALAR_PTR_LITERAL(195, 84, 31, 148, 26, 167, 194, 104)}};
static const lean_object* lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__5_value;
static lean_once_cell_t lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__6;
static const lean_string_object lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "FVarId"};
static const lean_object* lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__7_value;
static const lean_ctor_object lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__8_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__8_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__8_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__7_value),LEAN_SCALAR_PTR_LITERAL(134, 80, 170, 214, 218, 146, 55, 86)}};
static const lean_ctor_object lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__8_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_instToExprRaw__mathlib_toExpr___closed__3_value),LEAN_SCALAR_PTR_LITERAL(246, 212, 153, 136, 172, 214, 179, 96)}};
static const lean_object* lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__8_value;
static lean_once_cell_t lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__9;
static const lean_ctor_object lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__10_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__10_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__10_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__0_value),LEAN_SCALAR_PTR_LITERAL(84, 208, 74, 211, 93, 83, 88, 82)}};
static const lean_ctor_object lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__10_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_instToExprLevel__mathlib_toExpr___closed__16_value),LEAN_SCALAR_PTR_LITERAL(28, 197, 45, 187, 18, 219, 14, 58)}};
static const lean_object* lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__10_value;
static lean_once_cell_t lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__11_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__11;
static const lean_string_object lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "sort"};
static const lean_object* lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__12_value;
static const lean_ctor_object lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__13_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__13_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__13_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__0_value),LEAN_SCALAR_PTR_LITERAL(84, 208, 74, 211, 93, 83, 88, 82)}};
static const lean_ctor_object lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__13_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__12_value),LEAN_SCALAR_PTR_LITERAL(64, 95, 209, 188, 135, 1, 196, 95)}};
static const lean_object* lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__13_value;
static lean_once_cell_t lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__14_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__14;
static const lean_string_object lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "const"};
static const lean_object* lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__15 = (const lean_object*)&lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__15_value;
static const lean_ctor_object lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__16_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__16_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__16_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__0_value),LEAN_SCALAR_PTR_LITERAL(84, 208, 74, 211, 93, 83, 88, 82)}};
static const lean_ctor_object lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__16_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__15_value),LEAN_SCALAR_PTR_LITERAL(22, 248, 240, 94, 191, 251, 149, 49)}};
static const lean_object* lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__16 = (const lean_object*)&lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__16_value;
static lean_once_cell_t lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__17_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__17;
static lean_once_cell_t lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__18_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__18;
static lean_once_cell_t lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__19_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__19;
static const lean_string_object lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "app"};
static const lean_object* lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__20 = (const lean_object*)&lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__20_value;
static const lean_ctor_object lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__21_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__21_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__21_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__0_value),LEAN_SCALAR_PTR_LITERAL(84, 208, 74, 211, 93, 83, 88, 82)}};
static const lean_ctor_object lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__21_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__20_value),LEAN_SCALAR_PTR_LITERAL(134, 107, 4, 185, 254, 245, 50, 185)}};
static const lean_object* lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__21 = (const lean_object*)&lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__21_value;
static lean_once_cell_t lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__22_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__22;
static const lean_string_object lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "lam"};
static const lean_object* lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__23 = (const lean_object*)&lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__23_value;
static const lean_ctor_object lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__24_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__24_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__24_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__0_value),LEAN_SCALAR_PTR_LITERAL(84, 208, 74, 211, 93, 83, 88, 82)}};
static const lean_ctor_object lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__24_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__23_value),LEAN_SCALAR_PTR_LITERAL(156, 194, 121, 61, 219, 0, 202, 155)}};
static const lean_object* lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__24 = (const lean_object*)&lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__24_value;
static lean_once_cell_t lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__25_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__25;
static const lean_string_object lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__26_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "forallE"};
static const lean_object* lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__26 = (const lean_object*)&lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__26_value;
static const lean_ctor_object lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__27_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__27_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__27_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__0_value),LEAN_SCALAR_PTR_LITERAL(84, 208, 74, 211, 93, 83, 88, 82)}};
static const lean_ctor_object lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__27_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__27_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__26_value),LEAN_SCALAR_PTR_LITERAL(209, 174, 244, 115, 50, 19, 87, 122)}};
static const lean_object* lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__27 = (const lean_object*)&lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__27_value;
static lean_once_cell_t lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__28_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__28;
static const lean_string_object lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__29_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "letE"};
static const lean_object* lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__29 = (const lean_object*)&lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__29_value;
static const lean_ctor_object lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__30_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__30_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__30_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__0_value),LEAN_SCALAR_PTR_LITERAL(84, 208, 74, 211, 93, 83, 88, 82)}};
static const lean_ctor_object lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__30_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__30_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__29_value),LEAN_SCALAR_PTR_LITERAL(218, 165, 179, 210, 92, 162, 150, 56)}};
static const lean_object* lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__30 = (const lean_object*)&lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__30_value;
static lean_once_cell_t lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__31_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__31;
static const lean_string_object lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__32_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "lit"};
static const lean_object* lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__32 = (const lean_object*)&lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__32_value;
static const lean_ctor_object lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__33_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__33_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__33_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__0_value),LEAN_SCALAR_PTR_LITERAL(84, 208, 74, 211, 93, 83, 88, 82)}};
static const lean_ctor_object lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__33_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__33_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__32_value),LEAN_SCALAR_PTR_LITERAL(142, 45, 148, 16, 248, 234, 208, 241)}};
static const lean_object* lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__33 = (const lean_object*)&lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__33_value;
static lean_once_cell_t lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__34_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__34;
static const lean_string_object lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__35_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Literal"};
static const lean_object* lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__35 = (const lean_object*)&lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__35_value;
static const lean_string_object lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__36_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "natVal"};
static const lean_object* lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__36 = (const lean_object*)&lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__36_value;
static const lean_ctor_object lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__37_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__37_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__37_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__35_value),LEAN_SCALAR_PTR_LITERAL(39, 22, 220, 12, 129, 114, 43, 97)}};
static const lean_ctor_object lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__37_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__37_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__36_value),LEAN_SCALAR_PTR_LITERAL(64, 199, 201, 37, 137, 51, 1, 129)}};
static const lean_object* lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__37 = (const lean_object*)&lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__37_value;
static lean_once_cell_t lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__38_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__38;
static const lean_string_object lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__39_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "strVal"};
static const lean_object* lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__39 = (const lean_object*)&lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__39_value;
static const lean_ctor_object lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__40_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__40_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__40_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__35_value),LEAN_SCALAR_PTR_LITERAL(39, 22, 220, 12, 129, 114, 43, 97)}};
static const lean_ctor_object lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__40_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__40_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__39_value),LEAN_SCALAR_PTR_LITERAL(68, 214, 249, 146, 84, 160, 212, 27)}};
static const lean_object* lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__40 = (const lean_object*)&lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__40_value;
static lean_once_cell_t lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__41_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__41;
static const lean_string_object lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__42_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "mdata"};
static const lean_object* lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__42 = (const lean_object*)&lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__42_value;
static const lean_ctor_object lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__43_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__43_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__43_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__0_value),LEAN_SCALAR_PTR_LITERAL(84, 208, 74, 211, 93, 83, 88, 82)}};
static const lean_ctor_object lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__43_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__43_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__42_value),LEAN_SCALAR_PTR_LITERAL(32, 170, 73, 140, 82, 239, 68, 98)}};
static const lean_object* lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__43 = (const lean_object*)&lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__43_value;
static lean_once_cell_t lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__44_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__44;
static const lean_string_object lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__45_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "proj"};
static const lean_object* lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__45 = (const lean_object*)&lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__45_value;
static const lean_ctor_object lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__46_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__46_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__46_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__0_value),LEAN_SCALAR_PTR_LITERAL(84, 208, 74, 211, 93, 83, 88, 82)}};
static const lean_ctor_object lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__46_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__46_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__45_value),LEAN_SCALAR_PTR_LITERAL(164, 93, 179, 84, 156, 219, 121, 238)}};
static const lean_object* lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__46 = (const lean_object*)&lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__46_value;
static lean_once_cell_t lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__47_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__47;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr(lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_instToExprExpr__mathlib___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_instToExprExpr__mathlib___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_instToExprExpr__mathlib___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_instToExprExpr__mathlib___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_instToExprExpr__mathlib___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_instToExprExpr__mathlib___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__0_value),LEAN_SCALAR_PTR_LITERAL(84, 208, 74, 211, 93, 83, 88, 82)}};
static const lean_object* lp_mathlib_Mathlib_instToExprExpr__mathlib___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_instToExprExpr__mathlib___closed__1_value;
static lean_once_cell_t lp_mathlib_Mathlib_instToExprExpr__mathlib___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_instToExprExpr__mathlib___closed__2;
static lean_once_cell_t lp_mathlib_Mathlib_instToExprExpr__mathlib___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_instToExprExpr__mathlib___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_instToExprExpr__mathlib;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_instToExprULift__mathlib_toExpr___redArg(lean_object* v_inst_6_, lean_object* v_inst_7_, lean_object* v_inst_8_, lean_object* v_x_9_){
_start:
{
lean_object* v_toExpr_10_; lean_object* v_toTypeExpr_11_; lean_object* v___x_13_; uint8_t v_isShared_14_; uint8_t v_isSharedCheck_25_; 
v_toExpr_10_ = lean_ctor_get(v_inst_6_, 0);
v_toTypeExpr_11_ = lean_ctor_get(v_inst_6_, 1);
v_isSharedCheck_25_ = !lean_is_exclusive(v_inst_6_);
if (v_isSharedCheck_25_ == 0)
{
v___x_13_ = v_inst_6_;
v_isShared_14_ = v_isSharedCheck_25_;
goto v_resetjp_12_;
}
else
{
lean_inc(v_toTypeExpr_11_);
lean_inc(v_toExpr_10_);
lean_dec(v_inst_6_);
v___x_13_ = lean_box(0);
v_isShared_14_ = v_isSharedCheck_25_;
goto v_resetjp_12_;
}
v_resetjp_12_:
{
lean_object* v___x_15_; lean_object* v___x_16_; lean_object* v___x_18_; 
v___x_15_ = ((lean_object*)(lp_mathlib_Mathlib_instToExprULift__mathlib_toExpr___redArg___closed__2));
v___x_16_ = lean_box(0);
if (v_isShared_14_ == 0)
{
lean_ctor_set_tag(v___x_13_, 1);
lean_ctor_set(v___x_13_, 1, v___x_16_);
lean_ctor_set(v___x_13_, 0, v_inst_8_);
v___x_18_ = v___x_13_;
goto v_reusejp_17_;
}
else
{
lean_object* v_reuseFailAlloc_24_; 
v_reuseFailAlloc_24_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_24_, 0, v_inst_8_);
lean_ctor_set(v_reuseFailAlloc_24_, 1, v___x_16_);
v___x_18_ = v_reuseFailAlloc_24_;
goto v_reusejp_17_;
}
v_reusejp_17_:
{
lean_object* v___x_19_; lean_object* v___x_20_; lean_object* v___x_21_; lean_object* v___x_22_; lean_object* v___x_23_; 
v___x_19_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_19_, 0, v_inst_7_);
lean_ctor_set(v___x_19_, 1, v___x_18_);
v___x_20_ = l_Lean_Expr_const___override(v___x_15_, v___x_19_);
v___x_21_ = l_Lean_Expr_app___override(v___x_20_, v_toTypeExpr_11_);
v___x_22_ = lean_apply_1(v_toExpr_10_, v_x_9_);
v___x_23_ = l_Lean_Expr_app___override(v___x_21_, v___x_22_);
return v___x_23_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_instToExprULift__mathlib_toExpr(lean_object* v_00_u03b1_26_, lean_object* v_inst_27_, lean_object* v_inst_28_, lean_object* v_inst_29_, lean_object* v_x_30_){
_start:
{
lean_object* v___x_31_; 
v___x_31_ = lp_mathlib_Mathlib_instToExprULift__mathlib_toExpr___redArg(v_inst_27_, v_inst_28_, v_inst_29_, v_x_30_);
return v___x_31_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_instToExprULift__mathlib___redArg(lean_object* v_inst_34_, lean_object* v_inst_35_, lean_object* v_inst_36_){
_start:
{
lean_object* v_toTypeExpr_37_; lean_object* v___x_38_; lean_object* v___x_39_; lean_object* v___x_40_; lean_object* v___x_41_; lean_object* v___x_42_; lean_object* v___x_43_; lean_object* v___x_44_; lean_object* v___x_45_; 
v_toTypeExpr_37_ = lean_ctor_get(v_inst_34_, 1);
lean_inc_ref(v_toTypeExpr_37_);
lean_inc(v_inst_36_);
lean_inc(v_inst_35_);
v___x_38_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_instToExprULift__mathlib_toExpr), 5, 4);
lean_closure_set(v___x_38_, 0, lean_box(0));
lean_closure_set(v___x_38_, 1, v_inst_34_);
lean_closure_set(v___x_38_, 2, v_inst_35_);
lean_closure_set(v___x_38_, 3, v_inst_36_);
v___x_39_ = ((lean_object*)(lp_mathlib_Mathlib_instToExprULift__mathlib___redArg___closed__0));
v___x_40_ = lean_box(0);
v___x_41_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_41_, 0, v_inst_36_);
lean_ctor_set(v___x_41_, 1, v___x_40_);
v___x_42_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_42_, 0, v_inst_35_);
lean_ctor_set(v___x_42_, 1, v___x_41_);
v___x_43_ = l_Lean_Expr_const___override(v___x_39_, v___x_42_);
v___x_44_ = l_Lean_Expr_app___override(v___x_43_, v_toTypeExpr_37_);
v___x_45_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_45_, 0, v___x_38_);
lean_ctor_set(v___x_45_, 1, v___x_44_);
return v___x_45_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_instToExprULift__mathlib(lean_object* v_00_u03b1_46_, lean_object* v_inst_47_, lean_object* v_inst_48_, lean_object* v_inst_49_){
_start:
{
lean_object* v___x_50_; 
v___x_50_ = lp_mathlib_Mathlib_instToExprULift__mathlib___redArg(v_inst_47_, v_inst_48_, v_inst_49_);
return v___x_50_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_instToExprPUnitOfToLevel__mathlib___lam__0(lean_object* v_inst_56_, lean_object* v_x_57_){
_start:
{
lean_object* v___x_58_; lean_object* v___x_59_; lean_object* v___x_60_; lean_object* v___x_61_; lean_object* v___x_62_; 
v___x_58_ = ((lean_object*)(lp_mathlib_Mathlib_instToExprPUnitOfToLevel__mathlib___lam__0___closed__2));
v___x_59_ = l_Lean_Level_succ___override(v_inst_56_);
v___x_60_ = lean_box(0);
v___x_61_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_61_, 0, v___x_59_);
lean_ctor_set(v___x_61_, 1, v___x_60_);
v___x_62_ = l_Lean_mkConst(v___x_58_, v___x_61_);
return v___x_62_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_instToExprPUnitOfToLevel__mathlib(lean_object* v_inst_65_){
_start:
{
lean_object* v___f_66_; lean_object* v___x_67_; lean_object* v___x_68_; lean_object* v___x_69_; lean_object* v___x_70_; lean_object* v___x_71_; lean_object* v___x_72_; 
lean_inc(v_inst_65_);
v___f_66_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_instToExprPUnitOfToLevel__mathlib___lam__0), 2, 1);
lean_closure_set(v___f_66_, 0, v_inst_65_);
v___x_67_ = ((lean_object*)(lp_mathlib_Mathlib_instToExprPUnitOfToLevel__mathlib___closed__0));
v___x_68_ = l_Lean_Level_succ___override(v_inst_65_);
v___x_69_ = lean_box(0);
v___x_70_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_70_, 0, v___x_68_);
lean_ctor_set(v___x_70_, 1, v___x_69_);
v___x_71_ = l_Lean_mkConst(v___x_67_, v___x_70_);
v___x_72_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_72_, 0, v___f_66_);
lean_ctor_set(v___x_72_, 1, v___x_71_);
return v___x_72_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_instToExprRaw__mathlib_toExpr___closed__5(void){
_start:
{
lean_object* v___x_82_; lean_object* v___x_83_; lean_object* v___x_84_; 
v___x_82_ = lean_box(0);
v___x_83_ = ((lean_object*)(lp_mathlib_Mathlib_instToExprRaw__mathlib_toExpr___closed__4));
v___x_84_ = l_Lean_Expr_const___override(v___x_83_, v___x_82_);
return v___x_84_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_instToExprRaw__mathlib_toExpr(lean_object* v_x_85_){
_start:
{
lean_object* v___x_86_; lean_object* v___x_87_; lean_object* v___x_88_; 
v___x_86_ = lean_obj_once(&lp_mathlib_Mathlib_instToExprRaw__mathlib_toExpr___closed__5, &lp_mathlib_Mathlib_instToExprRaw__mathlib_toExpr___closed__5_once, _init_lp_mathlib_Mathlib_instToExprRaw__mathlib_toExpr___closed__5);
v___x_87_ = l_Lean_mkNatLit(v_x_85_);
v___x_88_ = l_Lean_Expr_app___override(v___x_86_, v___x_87_);
return v___x_88_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_instToExprRaw__mathlib___closed__2(void){
_start:
{
lean_object* v___x_94_; lean_object* v___x_95_; lean_object* v___x_96_; 
v___x_94_ = lean_box(0);
v___x_95_ = ((lean_object*)(lp_mathlib_Mathlib_instToExprRaw__mathlib___closed__1));
v___x_96_ = l_Lean_Expr_const___override(v___x_95_, v___x_94_);
return v___x_96_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_instToExprRaw__mathlib___closed__3(void){
_start:
{
lean_object* v___x_97_; lean_object* v___x_98_; lean_object* v___x_99_; 
v___x_97_ = lean_obj_once(&lp_mathlib_Mathlib_instToExprRaw__mathlib___closed__2, &lp_mathlib_Mathlib_instToExprRaw__mathlib___closed__2_once, _init_lp_mathlib_Mathlib_instToExprRaw__mathlib___closed__2);
v___x_98_ = ((lean_object*)(lp_mathlib_Mathlib_instToExprRaw__mathlib___closed__0));
v___x_99_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_99_, 0, v___x_98_);
lean_ctor_set(v___x_99_, 1, v___x_97_);
return v___x_99_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_instToExprRaw__mathlib(void){
_start:
{
lean_object* v___x_100_; 
v___x_100_ = lean_obj_once(&lp_mathlib_Mathlib_instToExprRaw__mathlib___closed__3, &lp_mathlib_Mathlib_instToExprRaw__mathlib___closed__3_once, _init_lp_mathlib_Mathlib_instToExprRaw__mathlib___closed__3);
return v___x_100_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_instToExprRaw__mathlib__1_toExpr___closed__2(void){
_start:
{
lean_object* v___x_106_; lean_object* v___x_107_; lean_object* v___x_108_; 
v___x_106_ = lean_box(0);
v___x_107_ = ((lean_object*)(lp_mathlib_Mathlib_instToExprRaw__mathlib__1_toExpr___closed__1));
v___x_108_ = l_Lean_Expr_const___override(v___x_107_, v___x_106_);
return v___x_108_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_instToExprRaw__mathlib__1_toExpr(lean_object* v_x_109_){
_start:
{
lean_object* v_str_110_; lean_object* v_startPos_111_; lean_object* v_stopPos_112_; lean_object* v___x_113_; lean_object* v___x_114_; lean_object* v___x_115_; lean_object* v___x_116_; lean_object* v___x_117_; lean_object* v___x_118_; lean_object* v___x_119_; 
v_str_110_ = lean_ctor_get(v_x_109_, 0);
lean_inc_ref(v_str_110_);
v_startPos_111_ = lean_ctor_get(v_x_109_, 1);
lean_inc(v_startPos_111_);
v_stopPos_112_ = lean_ctor_get(v_x_109_, 2);
lean_inc(v_stopPos_112_);
lean_dec_ref(v_x_109_);
v___x_113_ = lean_obj_once(&lp_mathlib_Mathlib_instToExprRaw__mathlib__1_toExpr___closed__2, &lp_mathlib_Mathlib_instToExprRaw__mathlib__1_toExpr___closed__2_once, _init_lp_mathlib_Mathlib_instToExprRaw__mathlib__1_toExpr___closed__2);
v___x_114_ = l_Lean_mkStrLit(v_str_110_);
v___x_115_ = l_Lean_Expr_app___override(v___x_113_, v___x_114_);
v___x_116_ = lp_mathlib_Mathlib_instToExprRaw__mathlib_toExpr(v_startPos_111_);
v___x_117_ = l_Lean_Expr_app___override(v___x_115_, v___x_116_);
v___x_118_ = lp_mathlib_Mathlib_instToExprRaw__mathlib_toExpr(v_stopPos_112_);
v___x_119_ = l_Lean_Expr_app___override(v___x_117_, v___x_118_);
return v___x_119_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_instToExprRaw__mathlib__1___closed__2(void){
_start:
{
lean_object* v___x_124_; lean_object* v___x_125_; lean_object* v___x_126_; 
v___x_124_ = lean_box(0);
v___x_125_ = ((lean_object*)(lp_mathlib_Mathlib_instToExprRaw__mathlib__1___closed__1));
v___x_126_ = l_Lean_Expr_const___override(v___x_125_, v___x_124_);
return v___x_126_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_instToExprRaw__mathlib__1___closed__3(void){
_start:
{
lean_object* v___x_127_; lean_object* v___x_128_; lean_object* v___x_129_; 
v___x_127_ = lean_obj_once(&lp_mathlib_Mathlib_instToExprRaw__mathlib__1___closed__2, &lp_mathlib_Mathlib_instToExprRaw__mathlib__1___closed__2_once, _init_lp_mathlib_Mathlib_instToExprRaw__mathlib__1___closed__2);
v___x_128_ = ((lean_object*)(lp_mathlib_Mathlib_instToExprRaw__mathlib__1___closed__0));
v___x_129_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_129_, 0, v___x_128_);
lean_ctor_set(v___x_129_, 1, v___x_127_);
return v___x_129_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_instToExprRaw__mathlib__1(void){
_start:
{
lean_object* v___x_130_; 
v___x_130_ = lean_obj_once(&lp_mathlib_Mathlib_instToExprRaw__mathlib__1___closed__3, &lp_mathlib_Mathlib_instToExprRaw__mathlib__1___closed__3_once, _init_lp_mathlib_Mathlib_instToExprRaw__mathlib__1___closed__3);
return v___x_130_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr___closed__4(void){
_start:
{
lean_object* v___x_138_; lean_object* v___x_139_; lean_object* v___x_140_; 
v___x_138_ = lean_box(0);
v___x_139_ = ((lean_object*)(lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr___closed__3));
v___x_140_ = l_Lean_Expr_const___override(v___x_139_, v___x_138_);
return v___x_140_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr___closed__7(void){
_start:
{
lean_object* v___x_146_; lean_object* v___x_147_; lean_object* v___x_148_; 
v___x_146_ = lean_box(0);
v___x_147_ = ((lean_object*)(lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr___closed__6));
v___x_148_ = l_Lean_Expr_const___override(v___x_147_, v___x_146_);
return v___x_148_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr___closed__11(void){
_start:
{
lean_object* v___x_154_; lean_object* v___x_155_; lean_object* v___x_156_; 
v___x_154_ = lean_box(0);
v___x_155_ = ((lean_object*)(lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr___closed__10));
v___x_156_ = l_Lean_mkConst(v___x_155_, v___x_154_);
return v___x_156_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr___closed__14(void){
_start:
{
lean_object* v___x_161_; lean_object* v___x_162_; lean_object* v___x_163_; 
v___x_161_ = lean_box(0);
v___x_162_ = ((lean_object*)(lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr___closed__13));
v___x_163_ = l_Lean_mkConst(v___x_162_, v___x_161_);
return v___x_163_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr___closed__17(void){
_start:
{
lean_object* v___x_169_; lean_object* v___x_170_; lean_object* v___x_171_; 
v___x_169_ = lean_box(0);
v___x_170_ = ((lean_object*)(lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr___closed__16));
v___x_171_ = l_Lean_Expr_const___override(v___x_170_, v___x_169_);
return v___x_171_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr(lean_object* v_x_172_){
_start:
{
switch(lean_obj_tag(v_x_172_))
{
case 0:
{
lean_object* v_leading_173_; lean_object* v_pos_174_; lean_object* v_trailing_175_; lean_object* v_endPos_176_; lean_object* v___x_177_; lean_object* v___x_178_; lean_object* v___x_179_; lean_object* v___x_180_; lean_object* v___x_181_; lean_object* v___x_182_; lean_object* v___x_183_; lean_object* v___x_184_; lean_object* v___x_185_; 
v_leading_173_ = lean_ctor_get(v_x_172_, 0);
lean_inc_ref(v_leading_173_);
v_pos_174_ = lean_ctor_get(v_x_172_, 1);
lean_inc(v_pos_174_);
v_trailing_175_ = lean_ctor_get(v_x_172_, 2);
lean_inc_ref(v_trailing_175_);
v_endPos_176_ = lean_ctor_get(v_x_172_, 3);
lean_inc(v_endPos_176_);
lean_dec_ref_known(v_x_172_, 4);
v___x_177_ = lean_obj_once(&lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr___closed__4, &lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr___closed__4_once, _init_lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr___closed__4);
v___x_178_ = lp_mathlib_Mathlib_instToExprRaw__mathlib__1_toExpr(v_leading_173_);
v___x_179_ = l_Lean_Expr_app___override(v___x_177_, v___x_178_);
v___x_180_ = lp_mathlib_Mathlib_instToExprRaw__mathlib_toExpr(v_pos_174_);
v___x_181_ = l_Lean_Expr_app___override(v___x_179_, v___x_180_);
v___x_182_ = lp_mathlib_Mathlib_instToExprRaw__mathlib__1_toExpr(v_trailing_175_);
v___x_183_ = l_Lean_Expr_app___override(v___x_181_, v___x_182_);
v___x_184_ = lp_mathlib_Mathlib_instToExprRaw__mathlib_toExpr(v_endPos_176_);
v___x_185_ = l_Lean_Expr_app___override(v___x_183_, v___x_184_);
return v___x_185_;
}
case 1:
{
lean_object* v_pos_186_; lean_object* v_endPos_187_; uint8_t v_canonical_188_; lean_object* v___x_189_; lean_object* v___x_190_; lean_object* v___x_191_; lean_object* v___x_192_; lean_object* v___x_193_; 
v_pos_186_ = lean_ctor_get(v_x_172_, 0);
lean_inc(v_pos_186_);
v_endPos_187_ = lean_ctor_get(v_x_172_, 1);
lean_inc(v_endPos_187_);
v_canonical_188_ = lean_ctor_get_uint8(v_x_172_, sizeof(void*)*2);
lean_dec_ref_known(v_x_172_, 2);
v___x_189_ = lean_obj_once(&lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr___closed__7, &lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr___closed__7_once, _init_lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr___closed__7);
v___x_190_ = lp_mathlib_Mathlib_instToExprRaw__mathlib_toExpr(v_pos_186_);
v___x_191_ = l_Lean_Expr_app___override(v___x_189_, v___x_190_);
v___x_192_ = lp_mathlib_Mathlib_instToExprRaw__mathlib_toExpr(v_endPos_187_);
v___x_193_ = l_Lean_Expr_app___override(v___x_191_, v___x_192_);
if (v_canonical_188_ == 0)
{
lean_object* v___x_194_; lean_object* v___x_195_; 
v___x_194_ = lean_obj_once(&lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr___closed__11, &lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr___closed__11_once, _init_lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr___closed__11);
v___x_195_ = l_Lean_Expr_app___override(v___x_193_, v___x_194_);
return v___x_195_;
}
else
{
lean_object* v___x_196_; lean_object* v___x_197_; 
v___x_196_ = lean_obj_once(&lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr___closed__14, &lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr___closed__14_once, _init_lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr___closed__14);
v___x_197_ = l_Lean_Expr_app___override(v___x_193_, v___x_196_);
return v___x_197_;
}
}
default: 
{
lean_object* v___x_198_; 
v___x_198_ = lean_obj_once(&lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr___closed__17, &lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr___closed__17_once, _init_lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr___closed__17);
return v___x_198_;
}
}
}
}
static lean_object* _init_lp_mathlib_Mathlib_instToExprSourceInfo__mathlib___closed__2(void){
_start:
{
lean_object* v___x_203_; lean_object* v___x_204_; lean_object* v___x_205_; 
v___x_203_ = lean_box(0);
v___x_204_ = ((lean_object*)(lp_mathlib_Mathlib_instToExprSourceInfo__mathlib___closed__1));
v___x_205_ = l_Lean_Expr_const___override(v___x_204_, v___x_203_);
return v___x_205_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_instToExprSourceInfo__mathlib___closed__3(void){
_start:
{
lean_object* v___x_206_; lean_object* v___x_207_; lean_object* v___x_208_; 
v___x_206_ = lean_obj_once(&lp_mathlib_Mathlib_instToExprSourceInfo__mathlib___closed__2, &lp_mathlib_Mathlib_instToExprSourceInfo__mathlib___closed__2_once, _init_lp_mathlib_Mathlib_instToExprSourceInfo__mathlib___closed__2);
v___x_207_ = ((lean_object*)(lp_mathlib_Mathlib_instToExprSourceInfo__mathlib___closed__0));
v___x_208_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_208_, 0, v___x_207_);
lean_ctor_set(v___x_208_, 1, v___x_206_);
return v___x_208_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_instToExprSourceInfo__mathlib(void){
_start:
{
lean_object* v___x_209_; 
v___x_209_ = lean_obj_once(&lp_mathlib_Mathlib_instToExprSourceInfo__mathlib___closed__3, &lp_mathlib_Mathlib_instToExprSourceInfo__mathlib___closed__3_once, _init_lp_mathlib_Mathlib_instToExprSourceInfo__mathlib___closed__3);
return v___x_209_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00__private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprSyntax__mathlib_toExpr_spec__1_spec__1(lean_object* v_nilFn_210_, lean_object* v_consFn_211_, lean_object* v_x_212_){
_start:
{
if (lean_obj_tag(v_x_212_) == 0)
{
lean_dec_ref(v_consFn_211_);
lean_inc_ref(v_nilFn_210_);
return v_nilFn_210_;
}
else
{
lean_object* v_head_213_; lean_object* v_tail_214_; lean_object* v___x_215_; lean_object* v___x_216_; lean_object* v___x_217_; 
v_head_213_ = lean_ctor_get(v_x_212_, 0);
lean_inc(v_head_213_);
v_tail_214_ = lean_ctor_get(v_x_212_, 1);
lean_inc(v_tail_214_);
lean_dec_ref_known(v_x_212_, 2);
v___x_215_ = l_Lean_mkStrLit(v_head_213_);
lean_inc_ref(v_consFn_211_);
v___x_216_ = lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00__private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprSyntax__mathlib_toExpr_spec__1_spec__1(v_nilFn_210_, v_consFn_211_, v_tail_214_);
v___x_217_ = l_Lean_mkAppB(v_consFn_211_, v___x_215_, v___x_216_);
return v___x_217_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00__private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprSyntax__mathlib_toExpr_spec__1_spec__1___boxed(lean_object* v_nilFn_218_, lean_object* v_consFn_219_, lean_object* v_x_220_){
_start:
{
lean_object* v_res_221_; 
v_res_221_ = lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00__private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprSyntax__mathlib_toExpr_spec__1_spec__1(v_nilFn_218_, v_consFn_219_, v_x_220_);
lean_dec_ref(v_nilFn_218_);
return v_res_221_;
}
}
static lean_object* _init_lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprSyntax__mathlib_toExpr_spec__1___closed__4(void){
_start:
{
lean_object* v___x_230_; lean_object* v___x_231_; lean_object* v___x_232_; 
v___x_230_ = lean_box(0);
v___x_231_ = ((lean_object*)(lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprSyntax__mathlib_toExpr_spec__1___closed__3));
v___x_232_ = l_Lean_Expr_const___override(v___x_231_, v___x_230_);
return v___x_232_;
}
}
static lean_object* _init_lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprSyntax__mathlib_toExpr_spec__1___closed__7(void){
_start:
{
lean_object* v___x_239_; lean_object* v___x_240_; lean_object* v___x_241_; 
v___x_239_ = lean_box(0);
v___x_240_ = ((lean_object*)(lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprSyntax__mathlib_toExpr_spec__1___closed__6));
v___x_241_ = l_Lean_Expr_const___override(v___x_240_, v___x_239_);
return v___x_241_;
}
}
static lean_object* _init_lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprSyntax__mathlib_toExpr_spec__1___closed__9(void){
_start:
{
lean_object* v___x_244_; lean_object* v___x_245_; lean_object* v_type_246_; 
v___x_244_ = lean_box(0);
v___x_245_ = ((lean_object*)(lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprSyntax__mathlib_toExpr_spec__1___closed__8));
v_type_246_ = l_Lean_mkConst(v___x_245_, v___x_244_);
return v_type_246_;
}
}
static lean_object* _init_lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprSyntax__mathlib_toExpr_spec__1___closed__14(void){
_start:
{
lean_object* v___x_255_; lean_object* v___x_256_; lean_object* v___x_257_; 
v___x_255_ = ((lean_object*)(lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprSyntax__mathlib_toExpr_spec__1___closed__13));
v___x_256_ = ((lean_object*)(lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprSyntax__mathlib_toExpr_spec__1___closed__12));
v___x_257_ = l_Lean_mkConst(v___x_256_, v___x_255_);
return v___x_257_;
}
}
static lean_object* _init_lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprSyntax__mathlib_toExpr_spec__1___closed__15(void){
_start:
{
lean_object* v_type_258_; lean_object* v___x_259_; lean_object* v_nil_260_; 
v_type_258_ = lean_obj_once(&lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprSyntax__mathlib_toExpr_spec__1___closed__9, &lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprSyntax__mathlib_toExpr_spec__1___closed__9_once, _init_lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprSyntax__mathlib_toExpr_spec__1___closed__9);
v___x_259_ = lean_obj_once(&lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprSyntax__mathlib_toExpr_spec__1___closed__14, &lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprSyntax__mathlib_toExpr_spec__1___closed__14_once, _init_lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprSyntax__mathlib_toExpr_spec__1___closed__14);
v_nil_260_ = l_Lean_Expr_app___override(v___x_259_, v_type_258_);
return v_nil_260_;
}
}
static lean_object* _init_lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprSyntax__mathlib_toExpr_spec__1___closed__18(void){
_start:
{
lean_object* v___x_265_; lean_object* v___x_266_; lean_object* v___x_267_; 
v___x_265_ = ((lean_object*)(lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprSyntax__mathlib_toExpr_spec__1___closed__13));
v___x_266_ = ((lean_object*)(lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprSyntax__mathlib_toExpr_spec__1___closed__17));
v___x_267_ = l_Lean_mkConst(v___x_266_, v___x_265_);
return v___x_267_;
}
}
static lean_object* _init_lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprSyntax__mathlib_toExpr_spec__1___closed__19(void){
_start:
{
lean_object* v_type_268_; lean_object* v___x_269_; lean_object* v_cons_270_; 
v_type_268_ = lean_obj_once(&lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprSyntax__mathlib_toExpr_spec__1___closed__9, &lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprSyntax__mathlib_toExpr_spec__1___closed__9_once, _init_lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprSyntax__mathlib_toExpr_spec__1___closed__9);
v___x_269_ = lean_obj_once(&lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprSyntax__mathlib_toExpr_spec__1___closed__18, &lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprSyntax__mathlib_toExpr_spec__1___closed__18_once, _init_lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprSyntax__mathlib_toExpr_spec__1___closed__18);
v_cons_270_ = l_Lean_Expr_app___override(v___x_269_, v_type_268_);
return v_cons_270_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprSyntax__mathlib_toExpr_spec__1(lean_object* v_nilFn_271_, lean_object* v_consFn_272_, lean_object* v_x_273_){
_start:
{
if (lean_obj_tag(v_x_273_) == 0)
{
lean_dec_ref(v_consFn_272_);
lean_inc_ref(v_nilFn_271_);
return v_nilFn_271_;
}
else
{
lean_object* v_head_274_; lean_object* v_tail_275_; lean_object* v___y_277_; 
v_head_274_ = lean_ctor_get(v_x_273_, 0);
lean_inc(v_head_274_);
v_tail_275_ = lean_ctor_get(v_x_273_, 1);
lean_inc(v_tail_275_);
lean_dec_ref_known(v_x_273_, 2);
if (lean_obj_tag(v_head_274_) == 0)
{
lean_object* v_ns_280_; lean_object* v___x_281_; lean_object* v___x_282_; lean_object* v___x_283_; 
v_ns_280_ = lean_ctor_get(v_head_274_, 0);
lean_inc(v_ns_280_);
lean_dec_ref_known(v_head_274_, 1);
v___x_281_ = lean_obj_once(&lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprSyntax__mathlib_toExpr_spec__1___closed__4, &lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprSyntax__mathlib_toExpr_spec__1___closed__4_once, _init_lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprSyntax__mathlib_toExpr_spec__1___closed__4);
v___x_282_ = l___private_Lean_ToExpr_0__Lean_Name_toExprAux(v_ns_280_);
v___x_283_ = l_Lean_Expr_app___override(v___x_281_, v___x_282_);
v___y_277_ = v___x_283_;
goto v___jp_276_;
}
else
{
lean_object* v_n_284_; lean_object* v_fields_285_; lean_object* v___x_286_; lean_object* v___x_287_; lean_object* v_nil_288_; lean_object* v_cons_289_; lean_object* v___x_290_; lean_object* v___x_291_; 
v_n_284_ = lean_ctor_get(v_head_274_, 0);
lean_inc(v_n_284_);
v_fields_285_ = lean_ctor_get(v_head_274_, 1);
lean_inc(v_fields_285_);
lean_dec_ref_known(v_head_274_, 2);
v___x_286_ = lean_obj_once(&lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprSyntax__mathlib_toExpr_spec__1___closed__7, &lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprSyntax__mathlib_toExpr_spec__1___closed__7_once, _init_lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprSyntax__mathlib_toExpr_spec__1___closed__7);
v___x_287_ = l___private_Lean_ToExpr_0__Lean_Name_toExprAux(v_n_284_);
v_nil_288_ = lean_obj_once(&lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprSyntax__mathlib_toExpr_spec__1___closed__15, &lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprSyntax__mathlib_toExpr_spec__1___closed__15_once, _init_lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprSyntax__mathlib_toExpr_spec__1___closed__15);
v_cons_289_ = lean_obj_once(&lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprSyntax__mathlib_toExpr_spec__1___closed__19, &lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprSyntax__mathlib_toExpr_spec__1___closed__19_once, _init_lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprSyntax__mathlib_toExpr_spec__1___closed__19);
v___x_290_ = lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00__private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprSyntax__mathlib_toExpr_spec__1_spec__1(v_nil_288_, v_cons_289_, v_fields_285_);
v___x_291_ = l_Lean_mkAppB(v___x_286_, v___x_287_, v___x_290_);
v___y_277_ = v___x_291_;
goto v___jp_276_;
}
v___jp_276_:
{
lean_object* v___x_278_; lean_object* v___x_279_; 
lean_inc_ref(v_consFn_272_);
v___x_278_ = lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprSyntax__mathlib_toExpr_spec__1(v_nilFn_271_, v_consFn_272_, v_tail_275_);
v___x_279_ = l_Lean_mkAppB(v_consFn_272_, v___y_277_, v___x_278_);
return v___x_279_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprSyntax__mathlib_toExpr_spec__1___boxed(lean_object* v_nilFn_292_, lean_object* v_consFn_293_, lean_object* v_x_294_){
_start:
{
lean_object* v_res_295_; 
v_res_295_ = lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprSyntax__mathlib_toExpr_spec__1(v_nilFn_292_, v_consFn_293_, v_x_294_);
lean_dec_ref(v_nilFn_292_);
return v_res_295_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_instToExprSyntax__mathlib_toExpr___closed__2(void){
_start:
{
lean_object* v___x_301_; lean_object* v___x_302_; lean_object* v___x_303_; 
v___x_301_ = lean_box(0);
v___x_302_ = ((lean_object*)(lp_mathlib_Mathlib_instToExprSyntax__mathlib_toExpr___closed__1));
v___x_303_ = l_Lean_Expr_const___override(v___x_302_, v___x_301_);
return v___x_303_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_instToExprSyntax__mathlib_toExpr___closed__4(void){
_start:
{
lean_object* v___x_307_; lean_object* v___x_308_; lean_object* v_type_309_; 
v___x_307_ = lean_box(0);
v___x_308_ = ((lean_object*)(lp_mathlib_Mathlib_instToExprSyntax__mathlib_toExpr___closed__3));
v_type_309_ = l_Lean_Expr_const___override(v___x_308_, v___x_307_);
return v_type_309_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_instToExprSyntax__mathlib_toExpr___closed__7(void){
_start:
{
lean_object* v___x_315_; lean_object* v___x_316_; lean_object* v___x_317_; 
v___x_315_ = lean_box(0);
v___x_316_ = ((lean_object*)(lp_mathlib_Mathlib_instToExprSyntax__mathlib_toExpr___closed__6));
v___x_317_ = l_Lean_Expr_const___override(v___x_316_, v___x_315_);
return v___x_317_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_instToExprSyntax__mathlib_toExpr___closed__10(void){
_start:
{
lean_object* v___x_322_; lean_object* v___x_323_; lean_object* v___x_324_; 
v___x_322_ = ((lean_object*)(lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprSyntax__mathlib_toExpr_spec__1___closed__13));
v___x_323_ = ((lean_object*)(lp_mathlib_Mathlib_instToExprSyntax__mathlib_toExpr___closed__9));
v___x_324_ = l_Lean_mkConst(v___x_323_, v___x_322_);
return v___x_324_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_instToExprSyntax__mathlib_toExpr___closed__11(void){
_start:
{
lean_object* v_type_325_; lean_object* v___x_326_; lean_object* v_nil_327_; 
v_type_325_ = lean_obj_once(&lp_mathlib_Mathlib_instToExprSyntax__mathlib_toExpr___closed__4, &lp_mathlib_Mathlib_instToExprSyntax__mathlib_toExpr___closed__4_once, _init_lp_mathlib_Mathlib_instToExprSyntax__mathlib_toExpr___closed__4);
v___x_326_ = lean_obj_once(&lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprSyntax__mathlib_toExpr_spec__1___closed__14, &lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprSyntax__mathlib_toExpr_spec__1___closed__14_once, _init_lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprSyntax__mathlib_toExpr_spec__1___closed__14);
v_nil_327_ = l_Lean_Expr_app___override(v___x_326_, v_type_325_);
return v_nil_327_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_instToExprSyntax__mathlib_toExpr___closed__12(void){
_start:
{
lean_object* v_type_328_; lean_object* v___x_329_; lean_object* v_cons_330_; 
v_type_328_ = lean_obj_once(&lp_mathlib_Mathlib_instToExprSyntax__mathlib_toExpr___closed__4, &lp_mathlib_Mathlib_instToExprSyntax__mathlib_toExpr___closed__4_once, _init_lp_mathlib_Mathlib_instToExprSyntax__mathlib_toExpr___closed__4);
v___x_329_ = lean_obj_once(&lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprSyntax__mathlib_toExpr_spec__1___closed__18, &lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprSyntax__mathlib_toExpr_spec__1___closed__18_once, _init_lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprSyntax__mathlib_toExpr_spec__1___closed__18);
v_cons_330_ = l_Lean_Expr_app___override(v___x_329_, v_type_328_);
return v_cons_330_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_instToExprSyntax__mathlib_toExpr___closed__15(void){
_start:
{
lean_object* v___x_336_; lean_object* v___x_337_; lean_object* v___x_338_; 
v___x_336_ = lean_box(0);
v___x_337_ = ((lean_object*)(lp_mathlib_Mathlib_instToExprSyntax__mathlib_toExpr___closed__14));
v___x_338_ = l_Lean_Expr_const___override(v___x_337_, v___x_336_);
return v___x_338_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_instToExprSyntax__mathlib_toExpr___closed__18(void){
_start:
{
lean_object* v___x_344_; lean_object* v___x_345_; lean_object* v___x_346_; 
v___x_344_ = lean_box(0);
v___x_345_ = ((lean_object*)(lp_mathlib_Mathlib_instToExprSyntax__mathlib_toExpr___closed__17));
v___x_346_ = l_Lean_Expr_const___override(v___x_345_, v___x_344_);
return v___x_346_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_instToExprSyntax__mathlib_toExpr___closed__20(void){
_start:
{
lean_object* v___x_351_; lean_object* v___x_352_; lean_object* v_type_353_; 
v___x_351_ = lean_box(0);
v___x_352_ = ((lean_object*)(lp_mathlib_Mathlib_instToExprSyntax__mathlib_toExpr___closed__19));
v_type_353_ = l_Lean_Expr_const___override(v___x_352_, v___x_351_);
return v_type_353_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_instToExprSyntax__mathlib_toExpr___closed__21(void){
_start:
{
lean_object* v_type_354_; lean_object* v___x_355_; lean_object* v_nil_356_; 
v_type_354_ = lean_obj_once(&lp_mathlib_Mathlib_instToExprSyntax__mathlib_toExpr___closed__20, &lp_mathlib_Mathlib_instToExprSyntax__mathlib_toExpr___closed__20_once, _init_lp_mathlib_Mathlib_instToExprSyntax__mathlib_toExpr___closed__20);
v___x_355_ = lean_obj_once(&lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprSyntax__mathlib_toExpr_spec__1___closed__14, &lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprSyntax__mathlib_toExpr_spec__1___closed__14_once, _init_lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprSyntax__mathlib_toExpr_spec__1___closed__14);
v_nil_356_ = l_Lean_Expr_app___override(v___x_355_, v_type_354_);
return v_nil_356_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_instToExprSyntax__mathlib_toExpr___closed__22(void){
_start:
{
lean_object* v_type_357_; lean_object* v___x_358_; lean_object* v_cons_359_; 
v_type_357_ = lean_obj_once(&lp_mathlib_Mathlib_instToExprSyntax__mathlib_toExpr___closed__20, &lp_mathlib_Mathlib_instToExprSyntax__mathlib_toExpr___closed__20_once, _init_lp_mathlib_Mathlib_instToExprSyntax__mathlib_toExpr___closed__20);
v___x_358_ = lean_obj_once(&lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprSyntax__mathlib_toExpr_spec__1___closed__18, &lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprSyntax__mathlib_toExpr_spec__1___closed__18_once, _init_lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprSyntax__mathlib_toExpr_spec__1___closed__18);
v_cons_359_ = l_Lean_Expr_app___override(v___x_358_, v_type_357_);
return v_cons_359_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_instToExprSyntax__mathlib_toExpr(lean_object* v_x_360_){
_start:
{
switch(lean_obj_tag(v_x_360_))
{
case 0:
{
lean_object* v___x_361_; 
v___x_361_ = lean_obj_once(&lp_mathlib_Mathlib_instToExprSyntax__mathlib_toExpr___closed__2, &lp_mathlib_Mathlib_instToExprSyntax__mathlib_toExpr___closed__2_once, _init_lp_mathlib_Mathlib_instToExprSyntax__mathlib_toExpr___closed__2);
return v___x_361_;
}
case 1:
{
lean_object* v_info_362_; lean_object* v_kind_363_; lean_object* v_args_364_; lean_object* v_type_365_; lean_object* v___x_366_; lean_object* v___x_367_; lean_object* v___x_368_; lean_object* v___x_369_; lean_object* v___x_370_; lean_object* v___x_371_; lean_object* v_nil_372_; lean_object* v_cons_373_; lean_object* v___x_374_; lean_object* v___x_375_; lean_object* v___x_376_; lean_object* v___x_377_; 
v_info_362_ = lean_ctor_get(v_x_360_, 0);
lean_inc(v_info_362_);
v_kind_363_ = lean_ctor_get(v_x_360_, 1);
lean_inc(v_kind_363_);
v_args_364_ = lean_ctor_get(v_x_360_, 2);
lean_inc_ref(v_args_364_);
lean_dec_ref_known(v_x_360_, 3);
v_type_365_ = lean_obj_once(&lp_mathlib_Mathlib_instToExprSyntax__mathlib_toExpr___closed__4, &lp_mathlib_Mathlib_instToExprSyntax__mathlib_toExpr___closed__4_once, _init_lp_mathlib_Mathlib_instToExprSyntax__mathlib_toExpr___closed__4);
v___x_366_ = lean_obj_once(&lp_mathlib_Mathlib_instToExprSyntax__mathlib_toExpr___closed__7, &lp_mathlib_Mathlib_instToExprSyntax__mathlib_toExpr___closed__7_once, _init_lp_mathlib_Mathlib_instToExprSyntax__mathlib_toExpr___closed__7);
v___x_367_ = lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr(v_info_362_);
v___x_368_ = l_Lean_Expr_app___override(v___x_366_, v___x_367_);
v___x_369_ = l___private_Lean_ToExpr_0__Lean_Name_toExprAux(v_kind_363_);
v___x_370_ = l_Lean_Expr_app___override(v___x_368_, v___x_369_);
v___x_371_ = lean_obj_once(&lp_mathlib_Mathlib_instToExprSyntax__mathlib_toExpr___closed__10, &lp_mathlib_Mathlib_instToExprSyntax__mathlib_toExpr___closed__10_once, _init_lp_mathlib_Mathlib_instToExprSyntax__mathlib_toExpr___closed__10);
v_nil_372_ = lean_obj_once(&lp_mathlib_Mathlib_instToExprSyntax__mathlib_toExpr___closed__11, &lp_mathlib_Mathlib_instToExprSyntax__mathlib_toExpr___closed__11_once, _init_lp_mathlib_Mathlib_instToExprSyntax__mathlib_toExpr___closed__11);
v_cons_373_ = lean_obj_once(&lp_mathlib_Mathlib_instToExprSyntax__mathlib_toExpr___closed__12, &lp_mathlib_Mathlib_instToExprSyntax__mathlib_toExpr___closed__12_once, _init_lp_mathlib_Mathlib_instToExprSyntax__mathlib_toExpr___closed__12);
v___x_374_ = lean_array_to_list(v_args_364_);
v___x_375_ = lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprSyntax__mathlib_toExpr_spec__0(v_nil_372_, v_cons_373_, v___x_374_);
v___x_376_ = l_Lean_mkAppB(v___x_371_, v_type_365_, v___x_375_);
v___x_377_ = l_Lean_Expr_app___override(v___x_370_, v___x_376_);
return v___x_377_;
}
case 2:
{
lean_object* v_info_378_; lean_object* v_val_379_; lean_object* v___x_380_; lean_object* v___x_381_; lean_object* v___x_382_; lean_object* v___x_383_; lean_object* v___x_384_; 
v_info_378_ = lean_ctor_get(v_x_360_, 0);
lean_inc(v_info_378_);
v_val_379_ = lean_ctor_get(v_x_360_, 1);
lean_inc_ref(v_val_379_);
lean_dec_ref_known(v_x_360_, 2);
v___x_380_ = lean_obj_once(&lp_mathlib_Mathlib_instToExprSyntax__mathlib_toExpr___closed__15, &lp_mathlib_Mathlib_instToExprSyntax__mathlib_toExpr___closed__15_once, _init_lp_mathlib_Mathlib_instToExprSyntax__mathlib_toExpr___closed__15);
v___x_381_ = lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr(v_info_378_);
v___x_382_ = l_Lean_Expr_app___override(v___x_380_, v___x_381_);
v___x_383_ = l_Lean_mkStrLit(v_val_379_);
v___x_384_ = l_Lean_Expr_app___override(v___x_382_, v___x_383_);
return v___x_384_;
}
default: 
{
lean_object* v_info_385_; lean_object* v_rawVal_386_; lean_object* v_val_387_; lean_object* v_preresolved_388_; lean_object* v___x_389_; lean_object* v___x_390_; lean_object* v___x_391_; lean_object* v___x_392_; lean_object* v___x_393_; lean_object* v___x_394_; lean_object* v___x_395_; lean_object* v_nil_396_; lean_object* v_cons_397_; lean_object* v___x_398_; lean_object* v___x_399_; 
v_info_385_ = lean_ctor_get(v_x_360_, 0);
lean_inc(v_info_385_);
v_rawVal_386_ = lean_ctor_get(v_x_360_, 1);
lean_inc_ref(v_rawVal_386_);
v_val_387_ = lean_ctor_get(v_x_360_, 2);
lean_inc(v_val_387_);
v_preresolved_388_ = lean_ctor_get(v_x_360_, 3);
lean_inc(v_preresolved_388_);
lean_dec_ref_known(v_x_360_, 4);
v___x_389_ = lean_obj_once(&lp_mathlib_Mathlib_instToExprSyntax__mathlib_toExpr___closed__18, &lp_mathlib_Mathlib_instToExprSyntax__mathlib_toExpr___closed__18_once, _init_lp_mathlib_Mathlib_instToExprSyntax__mathlib_toExpr___closed__18);
v___x_390_ = lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr(v_info_385_);
v___x_391_ = l_Lean_Expr_app___override(v___x_389_, v___x_390_);
v___x_392_ = lp_mathlib_Mathlib_instToExprRaw__mathlib__1_toExpr(v_rawVal_386_);
v___x_393_ = l_Lean_Expr_app___override(v___x_391_, v___x_392_);
v___x_394_ = l___private_Lean_ToExpr_0__Lean_Name_toExprAux(v_val_387_);
v___x_395_ = l_Lean_Expr_app___override(v___x_393_, v___x_394_);
v_nil_396_ = lean_obj_once(&lp_mathlib_Mathlib_instToExprSyntax__mathlib_toExpr___closed__21, &lp_mathlib_Mathlib_instToExprSyntax__mathlib_toExpr___closed__21_once, _init_lp_mathlib_Mathlib_instToExprSyntax__mathlib_toExpr___closed__21);
v_cons_397_ = lean_obj_once(&lp_mathlib_Mathlib_instToExprSyntax__mathlib_toExpr___closed__22, &lp_mathlib_Mathlib_instToExprSyntax__mathlib_toExpr___closed__22_once, _init_lp_mathlib_Mathlib_instToExprSyntax__mathlib_toExpr___closed__22);
v___x_398_ = lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprSyntax__mathlib_toExpr_spec__1(v_nil_396_, v_cons_397_, v_preresolved_388_);
v___x_399_ = l_Lean_Expr_app___override(v___x_395_, v___x_398_);
return v___x_399_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprSyntax__mathlib_toExpr_spec__0(lean_object* v_nilFn_400_, lean_object* v_consFn_401_, lean_object* v_x_402_){
_start:
{
if (lean_obj_tag(v_x_402_) == 0)
{
lean_dec_ref(v_consFn_401_);
lean_inc_ref(v_nilFn_400_);
return v_nilFn_400_;
}
else
{
lean_object* v_head_403_; lean_object* v_tail_404_; lean_object* v___x_405_; lean_object* v___x_406_; lean_object* v___x_407_; 
v_head_403_ = lean_ctor_get(v_x_402_, 0);
lean_inc(v_head_403_);
v_tail_404_ = lean_ctor_get(v_x_402_, 1);
lean_inc(v_tail_404_);
lean_dec_ref_known(v_x_402_, 2);
v___x_405_ = lp_mathlib_Mathlib_instToExprSyntax__mathlib_toExpr(v_head_403_);
lean_inc_ref(v_consFn_401_);
v___x_406_ = lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprSyntax__mathlib_toExpr_spec__0(v_nilFn_400_, v_consFn_401_, v_tail_404_);
v___x_407_ = l_Lean_mkAppB(v_consFn_401_, v___x_405_, v___x_406_);
return v___x_407_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprSyntax__mathlib_toExpr_spec__0___boxed(lean_object* v_nilFn_408_, lean_object* v_consFn_409_, lean_object* v_x_410_){
_start:
{
lean_object* v_res_411_; 
v_res_411_ = lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprSyntax__mathlib_toExpr_spec__0(v_nilFn_408_, v_consFn_409_, v_x_410_);
lean_dec_ref(v_nilFn_408_);
return v_res_411_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_instToExprSyntax__mathlib___closed__1(void){
_start:
{
lean_object* v___x_413_; lean_object* v___x_414_; lean_object* v___x_415_; 
v___x_413_ = lean_obj_once(&lp_mathlib_Mathlib_instToExprSyntax__mathlib_toExpr___closed__4, &lp_mathlib_Mathlib_instToExprSyntax__mathlib_toExpr___closed__4_once, _init_lp_mathlib_Mathlib_instToExprSyntax__mathlib_toExpr___closed__4);
v___x_414_ = ((lean_object*)(lp_mathlib_Mathlib_instToExprSyntax__mathlib___closed__0));
v___x_415_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_415_, 0, v___x_414_);
lean_ctor_set(v___x_415_, 1, v___x_413_);
return v___x_415_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_instToExprSyntax__mathlib(void){
_start:
{
lean_object* v___x_416_; 
v___x_416_ = lean_obj_once(&lp_mathlib_Mathlib_instToExprSyntax__mathlib___closed__1, &lp_mathlib_Mathlib_instToExprSyntax__mathlib___closed__1_once, _init_lp_mathlib_Mathlib_instToExprSyntax__mathlib___closed__1);
return v___x_416_;
}
}
static lean_object* _init_lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__3(void){
_start:
{
lean_object* v___x_423_; lean_object* v___x_424_; lean_object* v___x_425_; 
v___x_423_ = lean_box(0);
v___x_424_ = ((lean_object*)(lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__2));
v___x_425_ = l_Lean_mkConst(v___x_424_, v___x_423_);
return v___x_425_;
}
}
static lean_object* _init_lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__6(void){
_start:
{
lean_object* v___x_431_; lean_object* v___x_432_; lean_object* v___x_433_; 
v___x_431_ = lean_box(0);
v___x_432_ = ((lean_object*)(lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__5));
v___x_433_ = l_Lean_mkConst(v___x_432_, v___x_431_);
return v___x_433_;
}
}
static lean_object* _init_lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__9(void){
_start:
{
lean_object* v___x_439_; lean_object* v___x_440_; lean_object* v___x_441_; 
v___x_439_ = lean_box(0);
v___x_440_ = ((lean_object*)(lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__8));
v___x_441_ = l_Lean_mkConst(v___x_440_, v___x_439_);
return v___x_441_;
}
}
static lean_object* _init_lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__12(void){
_start:
{
lean_object* v___x_447_; lean_object* v___x_448_; lean_object* v___x_449_; 
v___x_447_ = lean_box(0);
v___x_448_ = ((lean_object*)(lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__11));
v___x_449_ = l_Lean_mkConst(v___x_448_, v___x_447_);
return v___x_449_;
}
}
static lean_object* _init_lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__15(void){
_start:
{
lean_object* v___x_455_; lean_object* v___x_456_; lean_object* v___x_457_; 
v___x_455_ = lean_box(0);
v___x_456_ = ((lean_object*)(lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__14));
v___x_457_ = l_Lean_mkConst(v___x_456_, v___x_455_);
return v___x_457_;
}
}
static lean_object* _init_lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__16(void){
_start:
{
lean_object* v___x_458_; lean_object* v___x_459_; 
v___x_458_ = lean_unsigned_to_nat(0u);
v___x_459_ = lean_nat_to_int(v___x_458_);
return v___x_459_;
}
}
static lean_object* _init_lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__20(void){
_start:
{
lean_object* v___x_465_; lean_object* v___x_466_; 
v___x_465_ = lean_unsigned_to_nat(0u);
v___x_466_ = l_Lean_Level_ofNat(v___x_465_);
return v___x_466_;
}
}
static lean_object* _init_lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__21(void){
_start:
{
lean_object* v___x_467_; lean_object* v___x_468_; lean_object* v___x_469_; 
v___x_467_ = lean_box(0);
v___x_468_ = lean_obj_once(&lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__20, &lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__20_once, _init_lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__20);
v___x_469_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_469_, 0, v___x_468_);
lean_ctor_set(v___x_469_, 1, v___x_467_);
return v___x_469_;
}
}
static lean_object* _init_lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__22(void){
_start:
{
lean_object* v___x_470_; lean_object* v___x_471_; lean_object* v___x_472_; 
v___x_470_ = lean_obj_once(&lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__21, &lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__21_once, _init_lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__21);
v___x_471_ = ((lean_object*)(lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__19));
v___x_472_ = l_Lean_Expr_const___override(v___x_471_, v___x_470_);
return v___x_472_;
}
}
static lean_object* _init_lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__25(void){
_start:
{
lean_object* v___x_476_; lean_object* v___x_477_; lean_object* v___x_478_; 
v___x_476_ = lean_box(0);
v___x_477_ = ((lean_object*)(lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__24));
v___x_478_ = l_Lean_Expr_const___override(v___x_477_, v___x_476_);
return v___x_478_;
}
}
static lean_object* _init_lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__28(void){
_start:
{
lean_object* v___x_483_; lean_object* v___x_484_; lean_object* v___x_485_; 
v___x_483_ = lean_box(0);
v___x_484_ = ((lean_object*)(lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__27));
v___x_485_ = l_Lean_Expr_const___override(v___x_484_, v___x_483_);
return v___x_485_;
}
}
static lean_object* _init_lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__31(void){
_start:
{
lean_object* v___x_491_; lean_object* v___x_492_; lean_object* v___x_493_; 
v___x_491_ = lean_box(0);
v___x_492_ = ((lean_object*)(lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__30));
v___x_493_ = l_Lean_mkConst(v___x_492_, v___x_491_);
return v___x_493_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg(lean_object* v_as_x27_494_, lean_object* v_b_495_){
_start:
{
if (lean_obj_tag(v_as_x27_494_) == 0)
{
return v_b_495_;
}
else
{
lean_object* v_head_496_; lean_object* v_tail_497_; lean_object* v_fst_498_; lean_object* v_snd_499_; lean_object* v___x_500_; 
v_head_496_ = lean_ctor_get(v_as_x27_494_, 0);
v_tail_497_ = lean_ctor_get(v_as_x27_494_, 1);
v_fst_498_ = lean_ctor_get(v_head_496_, 0);
v_snd_499_ = lean_ctor_get(v_head_496_, 1);
lean_inc(v_fst_498_);
v___x_500_ = l___private_Lean_ToExpr_0__Lean_Name_toExprAux(v_fst_498_);
switch(lean_obj_tag(v_snd_499_))
{
case 0:
{
lean_object* v_v_501_; lean_object* v___x_502_; lean_object* v___x_503_; lean_object* v___x_504_; 
v_v_501_ = lean_ctor_get(v_snd_499_, 0);
v___x_502_ = lean_obj_once(&lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__3, &lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__3_once, _init_lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__3);
lean_inc_ref(v_v_501_);
v___x_503_ = l_Lean_mkStrLit(v_v_501_);
v___x_504_ = l_Lean_mkApp3(v___x_502_, v_b_495_, v___x_500_, v___x_503_);
v_as_x27_494_ = v_tail_497_;
v_b_495_ = v___x_504_;
goto _start;
}
case 1:
{
uint8_t v_v_506_; lean_object* v___x_507_; 
v_v_506_ = lean_ctor_get_uint8(v_snd_499_, 0);
v___x_507_ = lean_obj_once(&lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__6, &lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__6_once, _init_lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__6);
if (v_v_506_ == 0)
{
lean_object* v___x_508_; lean_object* v___x_509_; 
v___x_508_ = lean_obj_once(&lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr___closed__11, &lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr___closed__11_once, _init_lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr___closed__11);
v___x_509_ = l_Lean_mkApp3(v___x_507_, v_b_495_, v___x_500_, v___x_508_);
v_as_x27_494_ = v_tail_497_;
v_b_495_ = v___x_509_;
goto _start;
}
else
{
lean_object* v___x_511_; lean_object* v___x_512_; 
v___x_511_ = lean_obj_once(&lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr___closed__14, &lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr___closed__14_once, _init_lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr___closed__14);
v___x_512_ = l_Lean_mkApp3(v___x_507_, v_b_495_, v___x_500_, v___x_511_);
v_as_x27_494_ = v_tail_497_;
v_b_495_ = v___x_512_;
goto _start;
}
}
case 2:
{
lean_object* v_v_514_; lean_object* v___x_515_; lean_object* v___x_516_; lean_object* v___x_517_; 
v_v_514_ = lean_ctor_get(v_snd_499_, 0);
v___x_515_ = lean_obj_once(&lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__9, &lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__9_once, _init_lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__9);
lean_inc(v_v_514_);
v___x_516_ = l___private_Lean_ToExpr_0__Lean_Name_toExprAux(v_v_514_);
v___x_517_ = l_Lean_mkApp3(v___x_515_, v_b_495_, v___x_500_, v___x_516_);
v_as_x27_494_ = v_tail_497_;
v_b_495_ = v___x_517_;
goto _start;
}
case 3:
{
lean_object* v_v_519_; lean_object* v___x_520_; lean_object* v___x_521_; lean_object* v___x_522_; 
v_v_519_ = lean_ctor_get(v_snd_499_, 0);
v___x_520_ = lean_obj_once(&lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__12, &lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__12_once, _init_lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__12);
lean_inc(v_v_519_);
v___x_521_ = l_Lean_mkNatLit(v_v_519_);
v___x_522_ = l_Lean_mkApp3(v___x_520_, v_b_495_, v___x_500_, v___x_521_);
v_as_x27_494_ = v_tail_497_;
v_b_495_ = v___x_522_;
goto _start;
}
case 4:
{
lean_object* v_v_524_; lean_object* v___x_525_; lean_object* v___x_526_; uint8_t v___x_527_; 
v_v_524_ = lean_ctor_get(v_snd_499_, 0);
v___x_525_ = lean_obj_once(&lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__15, &lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__15_once, _init_lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__15);
v___x_526_ = lean_obj_once(&lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__16, &lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__16_once, _init_lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__16);
v___x_527_ = lean_int_dec_le(v___x_526_, v_v_524_);
if (v___x_527_ == 0)
{
lean_object* v___x_528_; lean_object* v___x_529_; lean_object* v___x_530_; lean_object* v___x_531_; lean_object* v___x_532_; lean_object* v___x_533_; lean_object* v___x_534_; lean_object* v___x_535_; 
v___x_528_ = lean_obj_once(&lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__22, &lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__22_once, _init_lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__22);
v___x_529_ = lean_obj_once(&lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__25, &lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__25_once, _init_lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__25);
v___x_530_ = lean_obj_once(&lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__28, &lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__28_once, _init_lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__28);
v___x_531_ = lean_int_neg(v_v_524_);
v___x_532_ = l_Int_toNat(v___x_531_);
lean_dec(v___x_531_);
v___x_533_ = l_Lean_instToExprInt_mkNat(v___x_532_);
v___x_534_ = l_Lean_mkApp3(v___x_528_, v___x_529_, v___x_530_, v___x_533_);
v___x_535_ = l_Lean_mkApp3(v___x_525_, v_b_495_, v___x_500_, v___x_534_);
v_as_x27_494_ = v_tail_497_;
v_b_495_ = v___x_535_;
goto _start;
}
else
{
lean_object* v___x_537_; lean_object* v___x_538_; lean_object* v___x_539_; 
v___x_537_ = l_Int_toNat(v_v_524_);
v___x_538_ = l_Lean_instToExprInt_mkNat(v___x_537_);
v___x_539_ = l_Lean_mkApp3(v___x_525_, v_b_495_, v___x_500_, v___x_538_);
v_as_x27_494_ = v_tail_497_;
v_b_495_ = v___x_539_;
goto _start;
}
}
default: 
{
lean_object* v_v_541_; lean_object* v___x_542_; lean_object* v___x_543_; lean_object* v___x_544_; 
v_v_541_ = lean_ctor_get(v_snd_499_, 0);
v___x_542_ = lean_obj_once(&lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__31, &lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__31_once, _init_lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___closed__31);
lean_inc(v_v_541_);
v___x_543_ = lp_mathlib_Mathlib_instToExprSyntax__mathlib_toExpr(v_v_541_);
v___x_544_ = l_Lean_mkApp3(v___x_542_, v_b_495_, v___x_500_, v___x_543_);
v_as_x27_494_ = v_tail_497_;
v_b_495_ = v___x_544_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg___boxed(lean_object* v_as_x27_546_, lean_object* v_b_547_){
_start:
{
lean_object* v_res_548_; 
v_res_548_ = lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg(v_as_x27_546_, v_b_547_);
lean_dec(v_as_x27_546_);
return v_res_548_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData___closed__3(void){
_start:
{
lean_object* v___x_555_; lean_object* v___x_556_; lean_object* v_e_557_; 
v___x_555_ = lean_box(0);
v___x_556_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData___closed__2));
v_e_557_ = l_Lean_mkConst(v___x_556_, v___x_555_);
return v_e_557_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData(lean_object* v_md_558_){
_start:
{
lean_object* v_e_559_; lean_object* v___x_560_; 
v_e_559_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData___closed__3, &lp_mathlib___private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData___closed__3_once, _init_lp_mathlib___private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData___closed__3);
v___x_560_ = lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg(v_md_558_, v_e_559_);
return v___x_560_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData___boxed(lean_object* v_md_561_){
_start:
{
lean_object* v_res_562_; 
v_res_562_ = lp_mathlib___private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData(v_md_561_);
lean_dec(v_md_561_);
return v_res_562_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0(lean_object* v_as_563_, lean_object* v_as_x27_564_, lean_object* v_b_565_, lean_object* v_a_566_){
_start:
{
lean_object* v___x_567_; 
v___x_567_ = lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___redArg(v_as_x27_564_, v_b_565_);
return v___x_567_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0___boxed(lean_object* v_as_568_, lean_object* v_as_x27_569_, lean_object* v_b_570_, lean_object* v_a_571_){
_start:
{
lean_object* v_res_572_; 
v_res_572_ = lp_mathlib_List_forIn_x27_loop___at___00__private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData_spec__0(v_as_568_, v_as_x27_569_, v_b_570_, v_a_571_);
lean_dec(v_as_x27_569_);
lean_dec(v_as_568_);
return v_res_572_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_instToExprMData__mathlib___closed__2(void){
_start:
{
lean_object* v___x_577_; lean_object* v___x_578_; lean_object* v___x_579_; 
v___x_577_ = lean_box(0);
v___x_578_ = ((lean_object*)(lp_mathlib_Mathlib_instToExprMData__mathlib___closed__1));
v___x_579_ = l_Lean_mkConst(v___x_578_, v___x_577_);
return v___x_579_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_instToExprMData__mathlib___closed__3(void){
_start:
{
lean_object* v___x_580_; lean_object* v___x_581_; lean_object* v___x_582_; 
v___x_580_ = lean_obj_once(&lp_mathlib_Mathlib_instToExprMData__mathlib___closed__2, &lp_mathlib_Mathlib_instToExprMData__mathlib___closed__2_once, _init_lp_mathlib_Mathlib_instToExprMData__mathlib___closed__2);
v___x_581_ = ((lean_object*)(lp_mathlib_Mathlib_instToExprMData__mathlib___closed__0));
v___x_582_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_582_, 0, v___x_581_);
lean_ctor_set(v___x_582_, 1, v___x_580_);
return v___x_582_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_instToExprMData__mathlib(void){
_start:
{
lean_object* v___x_583_; 
v___x_583_ = lean_obj_once(&lp_mathlib_Mathlib_instToExprMData__mathlib___closed__3, &lp_mathlib_Mathlib_instToExprMData__mathlib___closed__3_once, _init_lp_mathlib_Mathlib_instToExprMData__mathlib___closed__3);
return v___x_583_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_instToExprMVarId__mathlib_toExpr___closed__2(void){
_start:
{
lean_object* v___x_589_; lean_object* v___x_590_; lean_object* v___x_591_; 
v___x_589_ = lean_box(0);
v___x_590_ = ((lean_object*)(lp_mathlib_Mathlib_instToExprMVarId__mathlib_toExpr___closed__1));
v___x_591_ = l_Lean_Expr_const___override(v___x_590_, v___x_589_);
return v___x_591_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_instToExprMVarId__mathlib_toExpr(lean_object* v_x_592_){
_start:
{
lean_object* v___x_593_; lean_object* v___x_594_; lean_object* v___x_595_; 
v___x_593_ = lean_obj_once(&lp_mathlib_Mathlib_instToExprMVarId__mathlib_toExpr___closed__2, &lp_mathlib_Mathlib_instToExprMVarId__mathlib_toExpr___closed__2_once, _init_lp_mathlib_Mathlib_instToExprMVarId__mathlib_toExpr___closed__2);
v___x_594_ = l___private_Lean_ToExpr_0__Lean_Name_toExprAux(v_x_592_);
v___x_595_ = l_Lean_Expr_app___override(v___x_593_, v___x_594_);
return v___x_595_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_instToExprMVarId__mathlib___closed__2(void){
_start:
{
lean_object* v___x_600_; lean_object* v___x_601_; lean_object* v___x_602_; 
v___x_600_ = lean_box(0);
v___x_601_ = ((lean_object*)(lp_mathlib_Mathlib_instToExprMVarId__mathlib___closed__1));
v___x_602_ = l_Lean_Expr_const___override(v___x_601_, v___x_600_);
return v___x_602_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_instToExprMVarId__mathlib___closed__3(void){
_start:
{
lean_object* v___x_603_; lean_object* v___x_604_; lean_object* v___x_605_; 
v___x_603_ = lean_obj_once(&lp_mathlib_Mathlib_instToExprMVarId__mathlib___closed__2, &lp_mathlib_Mathlib_instToExprMVarId__mathlib___closed__2_once, _init_lp_mathlib_Mathlib_instToExprMVarId__mathlib___closed__2);
v___x_604_ = ((lean_object*)(lp_mathlib_Mathlib_instToExprMVarId__mathlib___closed__0));
v___x_605_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_605_, 0, v___x_604_);
lean_ctor_set(v___x_605_, 1, v___x_603_);
return v___x_605_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_instToExprMVarId__mathlib(void){
_start:
{
lean_object* v___x_606_; 
v___x_606_ = lean_obj_once(&lp_mathlib_Mathlib_instToExprMVarId__mathlib___closed__3, &lp_mathlib_Mathlib_instToExprMVarId__mathlib___closed__3_once, _init_lp_mathlib_Mathlib_instToExprMVarId__mathlib___closed__3);
return v___x_606_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_instToExprLevelMVarId__mathlib_toExpr___closed__2(void){
_start:
{
lean_object* v___x_612_; lean_object* v___x_613_; lean_object* v___x_614_; 
v___x_612_ = lean_box(0);
v___x_613_ = ((lean_object*)(lp_mathlib_Mathlib_instToExprLevelMVarId__mathlib_toExpr___closed__1));
v___x_614_ = l_Lean_Expr_const___override(v___x_613_, v___x_612_);
return v___x_614_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_instToExprLevelMVarId__mathlib_toExpr(lean_object* v_x_615_){
_start:
{
lean_object* v___x_616_; lean_object* v___x_617_; lean_object* v___x_618_; 
v___x_616_ = lean_obj_once(&lp_mathlib_Mathlib_instToExprLevelMVarId__mathlib_toExpr___closed__2, &lp_mathlib_Mathlib_instToExprLevelMVarId__mathlib_toExpr___closed__2_once, _init_lp_mathlib_Mathlib_instToExprLevelMVarId__mathlib_toExpr___closed__2);
v___x_617_ = l___private_Lean_ToExpr_0__Lean_Name_toExprAux(v_x_615_);
v___x_618_ = l_Lean_Expr_app___override(v___x_616_, v___x_617_);
return v___x_618_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_instToExprLevelMVarId__mathlib___closed__2(void){
_start:
{
lean_object* v___x_623_; lean_object* v___x_624_; lean_object* v___x_625_; 
v___x_623_ = lean_box(0);
v___x_624_ = ((lean_object*)(lp_mathlib_Mathlib_instToExprLevelMVarId__mathlib___closed__1));
v___x_625_ = l_Lean_Expr_const___override(v___x_624_, v___x_623_);
return v___x_625_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_instToExprLevelMVarId__mathlib___closed__3(void){
_start:
{
lean_object* v___x_626_; lean_object* v___x_627_; lean_object* v___x_628_; 
v___x_626_ = lean_obj_once(&lp_mathlib_Mathlib_instToExprLevelMVarId__mathlib___closed__2, &lp_mathlib_Mathlib_instToExprLevelMVarId__mathlib___closed__2_once, _init_lp_mathlib_Mathlib_instToExprLevelMVarId__mathlib___closed__2);
v___x_627_ = ((lean_object*)(lp_mathlib_Mathlib_instToExprLevelMVarId__mathlib___closed__0));
v___x_628_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_628_, 0, v___x_627_);
lean_ctor_set(v___x_628_, 1, v___x_626_);
return v___x_628_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_instToExprLevelMVarId__mathlib(void){
_start:
{
lean_object* v___x_629_; 
v___x_629_ = lean_obj_once(&lp_mathlib_Mathlib_instToExprLevelMVarId__mathlib___closed__3, &lp_mathlib_Mathlib_instToExprLevelMVarId__mathlib___closed__3_once, _init_lp_mathlib_Mathlib_instToExprLevelMVarId__mathlib___closed__3);
return v___x_629_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_instToExprLevel__mathlib_toExpr___closed__3(void){
_start:
{
lean_object* v___x_636_; lean_object* v___x_637_; lean_object* v___x_638_; 
v___x_636_ = lean_box(0);
v___x_637_ = ((lean_object*)(lp_mathlib_Mathlib_instToExprLevel__mathlib_toExpr___closed__2));
v___x_638_ = l_Lean_Expr_const___override(v___x_637_, v___x_636_);
return v___x_638_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_instToExprLevel__mathlib_toExpr___closed__6(void){
_start:
{
lean_object* v___x_644_; lean_object* v___x_645_; lean_object* v___x_646_; 
v___x_644_ = lean_box(0);
v___x_645_ = ((lean_object*)(lp_mathlib_Mathlib_instToExprLevel__mathlib_toExpr___closed__5));
v___x_646_ = l_Lean_Expr_const___override(v___x_645_, v___x_644_);
return v___x_646_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_instToExprLevel__mathlib_toExpr___closed__9(void){
_start:
{
lean_object* v___x_652_; lean_object* v___x_653_; lean_object* v___x_654_; 
v___x_652_ = lean_box(0);
v___x_653_ = ((lean_object*)(lp_mathlib_Mathlib_instToExprLevel__mathlib_toExpr___closed__8));
v___x_654_ = l_Lean_Expr_const___override(v___x_653_, v___x_652_);
return v___x_654_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_instToExprLevel__mathlib_toExpr___closed__12(void){
_start:
{
lean_object* v___x_660_; lean_object* v___x_661_; lean_object* v___x_662_; 
v___x_660_ = lean_box(0);
v___x_661_ = ((lean_object*)(lp_mathlib_Mathlib_instToExprLevel__mathlib_toExpr___closed__11));
v___x_662_ = l_Lean_Expr_const___override(v___x_661_, v___x_660_);
return v___x_662_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_instToExprLevel__mathlib_toExpr___closed__15(void){
_start:
{
lean_object* v___x_668_; lean_object* v___x_669_; lean_object* v___x_670_; 
v___x_668_ = lean_box(0);
v___x_669_ = ((lean_object*)(lp_mathlib_Mathlib_instToExprLevel__mathlib_toExpr___closed__14));
v___x_670_ = l_Lean_Expr_const___override(v___x_669_, v___x_668_);
return v___x_670_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_instToExprLevel__mathlib_toExpr___closed__18(void){
_start:
{
lean_object* v___x_676_; lean_object* v___x_677_; lean_object* v___x_678_; 
v___x_676_ = lean_box(0);
v___x_677_ = ((lean_object*)(lp_mathlib_Mathlib_instToExprLevel__mathlib_toExpr___closed__17));
v___x_678_ = l_Lean_Expr_const___override(v___x_677_, v___x_676_);
return v___x_678_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_instToExprLevel__mathlib_toExpr(lean_object* v_x_679_){
_start:
{
switch(lean_obj_tag(v_x_679_))
{
case 0:
{
lean_object* v___x_680_; 
v___x_680_ = lean_obj_once(&lp_mathlib_Mathlib_instToExprLevel__mathlib_toExpr___closed__3, &lp_mathlib_Mathlib_instToExprLevel__mathlib_toExpr___closed__3_once, _init_lp_mathlib_Mathlib_instToExprLevel__mathlib_toExpr___closed__3);
return v___x_680_;
}
case 1:
{
lean_object* v_a_681_; lean_object* v___x_682_; lean_object* v___x_683_; lean_object* v___x_684_; 
v_a_681_ = lean_ctor_get(v_x_679_, 0);
lean_inc(v_a_681_);
lean_dec_ref_known(v_x_679_, 1);
v___x_682_ = lean_obj_once(&lp_mathlib_Mathlib_instToExprLevel__mathlib_toExpr___closed__6, &lp_mathlib_Mathlib_instToExprLevel__mathlib_toExpr___closed__6_once, _init_lp_mathlib_Mathlib_instToExprLevel__mathlib_toExpr___closed__6);
v___x_683_ = lp_mathlib_Mathlib_instToExprLevel__mathlib_toExpr(v_a_681_);
v___x_684_ = l_Lean_Expr_app___override(v___x_682_, v___x_683_);
return v___x_684_;
}
case 2:
{
lean_object* v_a_685_; lean_object* v_a_686_; lean_object* v___x_687_; lean_object* v___x_688_; lean_object* v___x_689_; lean_object* v___x_690_; lean_object* v___x_691_; 
v_a_685_ = lean_ctor_get(v_x_679_, 0);
lean_inc(v_a_685_);
v_a_686_ = lean_ctor_get(v_x_679_, 1);
lean_inc(v_a_686_);
lean_dec_ref_known(v_x_679_, 2);
v___x_687_ = lean_obj_once(&lp_mathlib_Mathlib_instToExprLevel__mathlib_toExpr___closed__9, &lp_mathlib_Mathlib_instToExprLevel__mathlib_toExpr___closed__9_once, _init_lp_mathlib_Mathlib_instToExprLevel__mathlib_toExpr___closed__9);
v___x_688_ = lp_mathlib_Mathlib_instToExprLevel__mathlib_toExpr(v_a_685_);
v___x_689_ = l_Lean_Expr_app___override(v___x_687_, v___x_688_);
v___x_690_ = lp_mathlib_Mathlib_instToExprLevel__mathlib_toExpr(v_a_686_);
v___x_691_ = l_Lean_Expr_app___override(v___x_689_, v___x_690_);
return v___x_691_;
}
case 3:
{
lean_object* v_a_692_; lean_object* v_a_693_; lean_object* v___x_694_; lean_object* v___x_695_; lean_object* v___x_696_; lean_object* v___x_697_; lean_object* v___x_698_; 
v_a_692_ = lean_ctor_get(v_x_679_, 0);
lean_inc(v_a_692_);
v_a_693_ = lean_ctor_get(v_x_679_, 1);
lean_inc(v_a_693_);
lean_dec_ref_known(v_x_679_, 2);
v___x_694_ = lean_obj_once(&lp_mathlib_Mathlib_instToExprLevel__mathlib_toExpr___closed__12, &lp_mathlib_Mathlib_instToExprLevel__mathlib_toExpr___closed__12_once, _init_lp_mathlib_Mathlib_instToExprLevel__mathlib_toExpr___closed__12);
v___x_695_ = lp_mathlib_Mathlib_instToExprLevel__mathlib_toExpr(v_a_692_);
v___x_696_ = l_Lean_Expr_app___override(v___x_694_, v___x_695_);
v___x_697_ = lp_mathlib_Mathlib_instToExprLevel__mathlib_toExpr(v_a_693_);
v___x_698_ = l_Lean_Expr_app___override(v___x_696_, v___x_697_);
return v___x_698_;
}
case 4:
{
lean_object* v_a_699_; lean_object* v___x_700_; lean_object* v___x_701_; lean_object* v___x_702_; 
v_a_699_ = lean_ctor_get(v_x_679_, 0);
lean_inc(v_a_699_);
lean_dec_ref_known(v_x_679_, 1);
v___x_700_ = lean_obj_once(&lp_mathlib_Mathlib_instToExprLevel__mathlib_toExpr___closed__15, &lp_mathlib_Mathlib_instToExprLevel__mathlib_toExpr___closed__15_once, _init_lp_mathlib_Mathlib_instToExprLevel__mathlib_toExpr___closed__15);
v___x_701_ = l___private_Lean_ToExpr_0__Lean_Name_toExprAux(v_a_699_);
v___x_702_ = l_Lean_Expr_app___override(v___x_700_, v___x_701_);
return v___x_702_;
}
default: 
{
lean_object* v_a_703_; lean_object* v___x_704_; lean_object* v___x_705_; lean_object* v___x_706_; 
v_a_703_ = lean_ctor_get(v_x_679_, 0);
lean_inc(v_a_703_);
lean_dec_ref_known(v_x_679_, 1);
v___x_704_ = lean_obj_once(&lp_mathlib_Mathlib_instToExprLevel__mathlib_toExpr___closed__18, &lp_mathlib_Mathlib_instToExprLevel__mathlib_toExpr___closed__18_once, _init_lp_mathlib_Mathlib_instToExprLevel__mathlib_toExpr___closed__18);
v___x_705_ = lp_mathlib_Mathlib_instToExprLevelMVarId__mathlib_toExpr(v_a_703_);
v___x_706_ = l_Lean_Expr_app___override(v___x_704_, v___x_705_);
return v___x_706_;
}
}
}
}
static lean_object* _init_lp_mathlib_Mathlib_instToExprLevel__mathlib___closed__2(void){
_start:
{
lean_object* v___x_711_; lean_object* v___x_712_; lean_object* v___x_713_; 
v___x_711_ = lean_box(0);
v___x_712_ = ((lean_object*)(lp_mathlib_Mathlib_instToExprLevel__mathlib___closed__1));
v___x_713_ = l_Lean_Expr_const___override(v___x_712_, v___x_711_);
return v___x_713_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_instToExprLevel__mathlib___closed__3(void){
_start:
{
lean_object* v___x_714_; lean_object* v___x_715_; lean_object* v___x_716_; 
v___x_714_ = lean_obj_once(&lp_mathlib_Mathlib_instToExprLevel__mathlib___closed__2, &lp_mathlib_Mathlib_instToExprLevel__mathlib___closed__2_once, _init_lp_mathlib_Mathlib_instToExprLevel__mathlib___closed__2);
v___x_715_ = ((lean_object*)(lp_mathlib_Mathlib_instToExprLevel__mathlib___closed__0));
v___x_716_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_716_, 0, v___x_715_);
lean_ctor_set(v___x_716_, 1, v___x_714_);
return v___x_716_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_instToExprLevel__mathlib(void){
_start:
{
lean_object* v___x_717_; 
v___x_717_ = lean_obj_once(&lp_mathlib_Mathlib_instToExprLevel__mathlib___closed__3, &lp_mathlib_Mathlib_instToExprLevel__mathlib___closed__3_once, _init_lp_mathlib_Mathlib_instToExprLevel__mathlib___closed__3);
return v___x_717_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_instToExprBinderInfo__mathlib_toExpr___closed__3(void){
_start:
{
lean_object* v___x_724_; lean_object* v___x_725_; lean_object* v___x_726_; 
v___x_724_ = lean_box(0);
v___x_725_ = ((lean_object*)(lp_mathlib_Mathlib_instToExprBinderInfo__mathlib_toExpr___closed__2));
v___x_726_ = l_Lean_Expr_const___override(v___x_725_, v___x_724_);
return v___x_726_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_instToExprBinderInfo__mathlib_toExpr___closed__6(void){
_start:
{
lean_object* v___x_732_; lean_object* v___x_733_; lean_object* v___x_734_; 
v___x_732_ = lean_box(0);
v___x_733_ = ((lean_object*)(lp_mathlib_Mathlib_instToExprBinderInfo__mathlib_toExpr___closed__5));
v___x_734_ = l_Lean_Expr_const___override(v___x_733_, v___x_732_);
return v___x_734_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_instToExprBinderInfo__mathlib_toExpr___closed__9(void){
_start:
{
lean_object* v___x_740_; lean_object* v___x_741_; lean_object* v___x_742_; 
v___x_740_ = lean_box(0);
v___x_741_ = ((lean_object*)(lp_mathlib_Mathlib_instToExprBinderInfo__mathlib_toExpr___closed__8));
v___x_742_ = l_Lean_Expr_const___override(v___x_741_, v___x_740_);
return v___x_742_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_instToExprBinderInfo__mathlib_toExpr___closed__12(void){
_start:
{
lean_object* v___x_748_; lean_object* v___x_749_; lean_object* v___x_750_; 
v___x_748_ = lean_box(0);
v___x_749_ = ((lean_object*)(lp_mathlib_Mathlib_instToExprBinderInfo__mathlib_toExpr___closed__11));
v___x_750_ = l_Lean_Expr_const___override(v___x_749_, v___x_748_);
return v___x_750_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_instToExprBinderInfo__mathlib_toExpr(uint8_t v_x_751_){
_start:
{
switch(v_x_751_)
{
case 0:
{
lean_object* v___x_752_; 
v___x_752_ = lean_obj_once(&lp_mathlib_Mathlib_instToExprBinderInfo__mathlib_toExpr___closed__3, &lp_mathlib_Mathlib_instToExprBinderInfo__mathlib_toExpr___closed__3_once, _init_lp_mathlib_Mathlib_instToExprBinderInfo__mathlib_toExpr___closed__3);
return v___x_752_;
}
case 1:
{
lean_object* v___x_753_; 
v___x_753_ = lean_obj_once(&lp_mathlib_Mathlib_instToExprBinderInfo__mathlib_toExpr___closed__6, &lp_mathlib_Mathlib_instToExprBinderInfo__mathlib_toExpr___closed__6_once, _init_lp_mathlib_Mathlib_instToExprBinderInfo__mathlib_toExpr___closed__6);
return v___x_753_;
}
case 2:
{
lean_object* v___x_754_; 
v___x_754_ = lean_obj_once(&lp_mathlib_Mathlib_instToExprBinderInfo__mathlib_toExpr___closed__9, &lp_mathlib_Mathlib_instToExprBinderInfo__mathlib_toExpr___closed__9_once, _init_lp_mathlib_Mathlib_instToExprBinderInfo__mathlib_toExpr___closed__9);
return v___x_754_;
}
default: 
{
lean_object* v___x_755_; 
v___x_755_ = lean_obj_once(&lp_mathlib_Mathlib_instToExprBinderInfo__mathlib_toExpr___closed__12, &lp_mathlib_Mathlib_instToExprBinderInfo__mathlib_toExpr___closed__12_once, _init_lp_mathlib_Mathlib_instToExprBinderInfo__mathlib_toExpr___closed__12);
return v___x_755_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_instToExprBinderInfo__mathlib_toExpr___boxed(lean_object* v_x_756_){
_start:
{
uint8_t v_x_145__boxed_757_; lean_object* v_res_758_; 
v_x_145__boxed_757_ = lean_unbox(v_x_756_);
v_res_758_ = lp_mathlib_Mathlib_instToExprBinderInfo__mathlib_toExpr(v_x_145__boxed_757_);
return v_res_758_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_instToExprBinderInfo__mathlib___closed__2(void){
_start:
{
lean_object* v___x_763_; lean_object* v___x_764_; lean_object* v___x_765_; 
v___x_763_ = lean_box(0);
v___x_764_ = ((lean_object*)(lp_mathlib_Mathlib_instToExprBinderInfo__mathlib___closed__1));
v___x_765_ = l_Lean_Expr_const___override(v___x_764_, v___x_763_);
return v___x_765_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_instToExprBinderInfo__mathlib___closed__3(void){
_start:
{
lean_object* v___x_766_; lean_object* v___x_767_; lean_object* v___x_768_; 
v___x_766_ = lean_obj_once(&lp_mathlib_Mathlib_instToExprBinderInfo__mathlib___closed__2, &lp_mathlib_Mathlib_instToExprBinderInfo__mathlib___closed__2_once, _init_lp_mathlib_Mathlib_instToExprBinderInfo__mathlib___closed__2);
v___x_767_ = ((lean_object*)(lp_mathlib_Mathlib_instToExprBinderInfo__mathlib___closed__0));
v___x_768_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_768_, 0, v___x_767_);
lean_ctor_set(v___x_768_, 1, v___x_766_);
return v___x_768_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_instToExprBinderInfo__mathlib(void){
_start:
{
lean_object* v___x_769_; 
v___x_769_ = lean_obj_once(&lp_mathlib_Mathlib_instToExprBinderInfo__mathlib___closed__3, &lp_mathlib_Mathlib_instToExprBinderInfo__mathlib___closed__3_once, _init_lp_mathlib_Mathlib_instToExprBinderInfo__mathlib___closed__3);
return v___x_769_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprExpr__mathlib_toExpr_spec__0(lean_object* v_nilFn_770_, lean_object* v_consFn_771_, lean_object* v_x_772_){
_start:
{
if (lean_obj_tag(v_x_772_) == 0)
{
lean_dec_ref(v_consFn_771_);
lean_inc_ref(v_nilFn_770_);
return v_nilFn_770_;
}
else
{
lean_object* v_head_773_; lean_object* v_tail_774_; lean_object* v___x_775_; lean_object* v___x_776_; lean_object* v___x_777_; 
v_head_773_ = lean_ctor_get(v_x_772_, 0);
lean_inc(v_head_773_);
v_tail_774_ = lean_ctor_get(v_x_772_, 1);
lean_inc(v_tail_774_);
lean_dec_ref_known(v_x_772_, 2);
v___x_775_ = lp_mathlib_Mathlib_instToExprLevel__mathlib_toExpr(v_head_773_);
lean_inc_ref(v_consFn_771_);
v___x_776_ = lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprExpr__mathlib_toExpr_spec__0(v_nilFn_770_, v_consFn_771_, v_tail_774_);
v___x_777_ = l_Lean_mkAppB(v_consFn_771_, v___x_775_, v___x_776_);
return v___x_777_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprExpr__mathlib_toExpr_spec__0___boxed(lean_object* v_nilFn_778_, lean_object* v_consFn_779_, lean_object* v_x_780_){
_start:
{
lean_object* v_res_781_; 
v_res_781_ = lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprExpr__mathlib_toExpr_spec__0(v_nilFn_778_, v_consFn_779_, v_x_780_);
lean_dec_ref(v_nilFn_778_);
return v_res_781_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__3(void){
_start:
{
lean_object* v___x_788_; lean_object* v___x_789_; lean_object* v___x_790_; 
v___x_788_ = lean_box(0);
v___x_789_ = ((lean_object*)(lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__2));
v___x_790_ = l_Lean_Expr_const___override(v___x_789_, v___x_788_);
return v___x_790_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__6(void){
_start:
{
lean_object* v___x_796_; lean_object* v___x_797_; lean_object* v___x_798_; 
v___x_796_ = lean_box(0);
v___x_797_ = ((lean_object*)(lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__5));
v___x_798_ = l_Lean_Expr_const___override(v___x_797_, v___x_796_);
return v___x_798_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__9(void){
_start:
{
lean_object* v___x_804_; lean_object* v___x_805_; lean_object* v___x_806_; 
v___x_804_ = lean_box(0);
v___x_805_ = ((lean_object*)(lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__8));
v___x_806_ = l_Lean_mkConst(v___x_805_, v___x_804_);
return v___x_806_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__11(void){
_start:
{
lean_object* v___x_811_; lean_object* v___x_812_; lean_object* v___x_813_; 
v___x_811_ = lean_box(0);
v___x_812_ = ((lean_object*)(lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__10));
v___x_813_ = l_Lean_Expr_const___override(v___x_812_, v___x_811_);
return v___x_813_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__14(void){
_start:
{
lean_object* v___x_819_; lean_object* v___x_820_; lean_object* v___x_821_; 
v___x_819_ = lean_box(0);
v___x_820_ = ((lean_object*)(lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__13));
v___x_821_ = l_Lean_Expr_const___override(v___x_820_, v___x_819_);
return v___x_821_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__17(void){
_start:
{
lean_object* v___x_827_; lean_object* v___x_828_; lean_object* v___x_829_; 
v___x_827_ = lean_box(0);
v___x_828_ = ((lean_object*)(lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__16));
v___x_829_ = l_Lean_Expr_const___override(v___x_828_, v___x_827_);
return v___x_829_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__18(void){
_start:
{
lean_object* v_type_830_; lean_object* v___x_831_; lean_object* v_nil_832_; 
v_type_830_ = lean_obj_once(&lp_mathlib_Mathlib_instToExprLevel__mathlib___closed__2, &lp_mathlib_Mathlib_instToExprLevel__mathlib___closed__2_once, _init_lp_mathlib_Mathlib_instToExprLevel__mathlib___closed__2);
v___x_831_ = lean_obj_once(&lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprSyntax__mathlib_toExpr_spec__1___closed__14, &lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprSyntax__mathlib_toExpr_spec__1___closed__14_once, _init_lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprSyntax__mathlib_toExpr_spec__1___closed__14);
v_nil_832_ = l_Lean_Expr_app___override(v___x_831_, v_type_830_);
return v_nil_832_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__19(void){
_start:
{
lean_object* v_type_833_; lean_object* v___x_834_; lean_object* v_cons_835_; 
v_type_833_ = lean_obj_once(&lp_mathlib_Mathlib_instToExprLevel__mathlib___closed__2, &lp_mathlib_Mathlib_instToExprLevel__mathlib___closed__2_once, _init_lp_mathlib_Mathlib_instToExprLevel__mathlib___closed__2);
v___x_834_ = lean_obj_once(&lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprSyntax__mathlib_toExpr_spec__1___closed__18, &lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprSyntax__mathlib_toExpr_spec__1___closed__18_once, _init_lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprSyntax__mathlib_toExpr_spec__1___closed__18);
v_cons_835_ = l_Lean_Expr_app___override(v___x_834_, v_type_833_);
return v_cons_835_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__22(void){
_start:
{
lean_object* v___x_841_; lean_object* v___x_842_; lean_object* v___x_843_; 
v___x_841_ = lean_box(0);
v___x_842_ = ((lean_object*)(lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__21));
v___x_843_ = l_Lean_Expr_const___override(v___x_842_, v___x_841_);
return v___x_843_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__25(void){
_start:
{
lean_object* v___x_849_; lean_object* v___x_850_; lean_object* v___x_851_; 
v___x_849_ = lean_box(0);
v___x_850_ = ((lean_object*)(lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__24));
v___x_851_ = l_Lean_Expr_const___override(v___x_850_, v___x_849_);
return v___x_851_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__28(void){
_start:
{
lean_object* v___x_857_; lean_object* v___x_858_; lean_object* v___x_859_; 
v___x_857_ = lean_box(0);
v___x_858_ = ((lean_object*)(lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__27));
v___x_859_ = l_Lean_Expr_const___override(v___x_858_, v___x_857_);
return v___x_859_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__31(void){
_start:
{
lean_object* v___x_865_; lean_object* v___x_866_; lean_object* v___x_867_; 
v___x_865_ = lean_box(0);
v___x_866_ = ((lean_object*)(lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__30));
v___x_867_ = l_Lean_Expr_const___override(v___x_866_, v___x_865_);
return v___x_867_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__34(void){
_start:
{
lean_object* v___x_873_; lean_object* v___x_874_; lean_object* v___x_875_; 
v___x_873_ = lean_box(0);
v___x_874_ = ((lean_object*)(lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__33));
v___x_875_ = l_Lean_Expr_const___override(v___x_874_, v___x_873_);
return v___x_875_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__38(void){
_start:
{
lean_object* v___x_882_; lean_object* v___x_883_; lean_object* v___x_884_; 
v___x_882_ = lean_box(0);
v___x_883_ = ((lean_object*)(lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__37));
v___x_884_ = l_Lean_mkConst(v___x_883_, v___x_882_);
return v___x_884_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__41(void){
_start:
{
lean_object* v___x_890_; lean_object* v___x_891_; lean_object* v___x_892_; 
v___x_890_ = lean_box(0);
v___x_891_ = ((lean_object*)(lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__40));
v___x_892_ = l_Lean_mkConst(v___x_891_, v___x_890_);
return v___x_892_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__44(void){
_start:
{
lean_object* v___x_898_; lean_object* v___x_899_; lean_object* v___x_900_; 
v___x_898_ = lean_box(0);
v___x_899_ = ((lean_object*)(lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__43));
v___x_900_ = l_Lean_Expr_const___override(v___x_899_, v___x_898_);
return v___x_900_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__47(void){
_start:
{
lean_object* v___x_906_; lean_object* v___x_907_; lean_object* v___x_908_; 
v___x_906_ = lean_box(0);
v___x_907_ = ((lean_object*)(lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__46));
v___x_908_ = l_Lean_Expr_const___override(v___x_907_, v___x_906_);
return v___x_908_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr(lean_object* v_x_909_){
_start:
{
switch(lean_obj_tag(v_x_909_))
{
case 0:
{
lean_object* v_deBruijnIndex_910_; lean_object* v___x_911_; lean_object* v___x_912_; lean_object* v___x_913_; 
v_deBruijnIndex_910_ = lean_ctor_get(v_x_909_, 0);
lean_inc(v_deBruijnIndex_910_);
lean_dec_ref_known(v_x_909_, 1);
v___x_911_ = lean_obj_once(&lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__3, &lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__3_once, _init_lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__3);
v___x_912_ = l_Lean_mkNatLit(v_deBruijnIndex_910_);
v___x_913_ = l_Lean_Expr_app___override(v___x_911_, v___x_912_);
return v___x_913_;
}
case 1:
{
lean_object* v_fvarId_914_; lean_object* v___x_915_; lean_object* v___x_916_; lean_object* v___x_917_; lean_object* v___x_918_; lean_object* v___x_919_; 
v_fvarId_914_ = lean_ctor_get(v_x_909_, 0);
lean_inc(v_fvarId_914_);
lean_dec_ref_known(v_x_909_, 1);
v___x_915_ = lean_obj_once(&lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__6, &lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__6_once, _init_lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__6);
v___x_916_ = lean_obj_once(&lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__9, &lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__9_once, _init_lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__9);
v___x_917_ = l___private_Lean_ToExpr_0__Lean_Name_toExprAux(v_fvarId_914_);
v___x_918_ = l_Lean_Expr_app___override(v___x_916_, v___x_917_);
v___x_919_ = l_Lean_Expr_app___override(v___x_915_, v___x_918_);
return v___x_919_;
}
case 2:
{
lean_object* v_mvarId_920_; lean_object* v___x_921_; lean_object* v___x_922_; lean_object* v___x_923_; 
v_mvarId_920_ = lean_ctor_get(v_x_909_, 0);
lean_inc(v_mvarId_920_);
lean_dec_ref_known(v_x_909_, 1);
v___x_921_ = lean_obj_once(&lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__11, &lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__11_once, _init_lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__11);
v___x_922_ = lp_mathlib_Mathlib_instToExprMVarId__mathlib_toExpr(v_mvarId_920_);
v___x_923_ = l_Lean_Expr_app___override(v___x_921_, v___x_922_);
return v___x_923_;
}
case 3:
{
lean_object* v_u_924_; lean_object* v___x_925_; lean_object* v___x_926_; lean_object* v___x_927_; 
v_u_924_ = lean_ctor_get(v_x_909_, 0);
lean_inc(v_u_924_);
lean_dec_ref_known(v_x_909_, 1);
v___x_925_ = lean_obj_once(&lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__14, &lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__14_once, _init_lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__14);
v___x_926_ = lp_mathlib_Mathlib_instToExprLevel__mathlib_toExpr(v_u_924_);
v___x_927_ = l_Lean_Expr_app___override(v___x_925_, v___x_926_);
return v___x_927_;
}
case 4:
{
lean_object* v_declName_928_; lean_object* v_us_929_; lean_object* v___x_930_; lean_object* v___x_931_; lean_object* v___x_932_; lean_object* v_nil_933_; lean_object* v_cons_934_; lean_object* v___x_935_; lean_object* v___x_936_; 
v_declName_928_ = lean_ctor_get(v_x_909_, 0);
lean_inc(v_declName_928_);
v_us_929_ = lean_ctor_get(v_x_909_, 1);
lean_inc(v_us_929_);
lean_dec_ref_known(v_x_909_, 2);
v___x_930_ = lean_obj_once(&lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__17, &lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__17_once, _init_lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__17);
v___x_931_ = l___private_Lean_ToExpr_0__Lean_Name_toExprAux(v_declName_928_);
v___x_932_ = l_Lean_Expr_app___override(v___x_930_, v___x_931_);
v_nil_933_ = lean_obj_once(&lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__18, &lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__18_once, _init_lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__18);
v_cons_934_ = lean_obj_once(&lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__19, &lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__19_once, _init_lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__19);
v___x_935_ = lp_mathlib___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00Mathlib_instToExprExpr__mathlib_toExpr_spec__0(v_nil_933_, v_cons_934_, v_us_929_);
v___x_936_ = l_Lean_Expr_app___override(v___x_932_, v___x_935_);
return v___x_936_;
}
case 5:
{
lean_object* v_fn_937_; lean_object* v_arg_938_; lean_object* v___x_939_; lean_object* v___x_940_; lean_object* v___x_941_; lean_object* v___x_942_; lean_object* v___x_943_; 
v_fn_937_ = lean_ctor_get(v_x_909_, 0);
lean_inc_ref(v_fn_937_);
v_arg_938_ = lean_ctor_get(v_x_909_, 1);
lean_inc_ref(v_arg_938_);
lean_dec_ref_known(v_x_909_, 2);
v___x_939_ = lean_obj_once(&lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__22, &lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__22_once, _init_lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__22);
v___x_940_ = lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr(v_fn_937_);
v___x_941_ = l_Lean_Expr_app___override(v___x_939_, v___x_940_);
v___x_942_ = lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr(v_arg_938_);
v___x_943_ = l_Lean_Expr_app___override(v___x_941_, v___x_942_);
return v___x_943_;
}
case 6:
{
lean_object* v_binderName_944_; lean_object* v_binderType_945_; lean_object* v_body_946_; uint8_t v_binderInfo_947_; lean_object* v___x_948_; lean_object* v___x_949_; lean_object* v___x_950_; lean_object* v___x_951_; lean_object* v___x_952_; lean_object* v___x_953_; lean_object* v___x_954_; lean_object* v___x_955_; lean_object* v___x_956_; 
v_binderName_944_ = lean_ctor_get(v_x_909_, 0);
lean_inc(v_binderName_944_);
v_binderType_945_ = lean_ctor_get(v_x_909_, 1);
lean_inc_ref(v_binderType_945_);
v_body_946_ = lean_ctor_get(v_x_909_, 2);
lean_inc_ref(v_body_946_);
v_binderInfo_947_ = lean_ctor_get_uint8(v_x_909_, sizeof(void*)*3 + 8);
lean_dec_ref_known(v_x_909_, 3);
v___x_948_ = lean_obj_once(&lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__25, &lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__25_once, _init_lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__25);
v___x_949_ = l___private_Lean_ToExpr_0__Lean_Name_toExprAux(v_binderName_944_);
v___x_950_ = l_Lean_Expr_app___override(v___x_948_, v___x_949_);
v___x_951_ = lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr(v_binderType_945_);
v___x_952_ = l_Lean_Expr_app___override(v___x_950_, v___x_951_);
v___x_953_ = lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr(v_body_946_);
v___x_954_ = l_Lean_Expr_app___override(v___x_952_, v___x_953_);
v___x_955_ = lp_mathlib_Mathlib_instToExprBinderInfo__mathlib_toExpr(v_binderInfo_947_);
v___x_956_ = l_Lean_Expr_app___override(v___x_954_, v___x_955_);
return v___x_956_;
}
case 7:
{
lean_object* v_binderName_957_; lean_object* v_binderType_958_; lean_object* v_body_959_; uint8_t v_binderInfo_960_; lean_object* v___x_961_; lean_object* v___x_962_; lean_object* v___x_963_; lean_object* v___x_964_; lean_object* v___x_965_; lean_object* v___x_966_; lean_object* v___x_967_; lean_object* v___x_968_; lean_object* v___x_969_; 
v_binderName_957_ = lean_ctor_get(v_x_909_, 0);
lean_inc(v_binderName_957_);
v_binderType_958_ = lean_ctor_get(v_x_909_, 1);
lean_inc_ref(v_binderType_958_);
v_body_959_ = lean_ctor_get(v_x_909_, 2);
lean_inc_ref(v_body_959_);
v_binderInfo_960_ = lean_ctor_get_uint8(v_x_909_, sizeof(void*)*3 + 8);
lean_dec_ref_known(v_x_909_, 3);
v___x_961_ = lean_obj_once(&lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__28, &lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__28_once, _init_lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__28);
v___x_962_ = l___private_Lean_ToExpr_0__Lean_Name_toExprAux(v_binderName_957_);
v___x_963_ = l_Lean_Expr_app___override(v___x_961_, v___x_962_);
v___x_964_ = lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr(v_binderType_958_);
v___x_965_ = l_Lean_Expr_app___override(v___x_963_, v___x_964_);
v___x_966_ = lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr(v_body_959_);
v___x_967_ = l_Lean_Expr_app___override(v___x_965_, v___x_966_);
v___x_968_ = lp_mathlib_Mathlib_instToExprBinderInfo__mathlib_toExpr(v_binderInfo_960_);
v___x_969_ = l_Lean_Expr_app___override(v___x_967_, v___x_968_);
return v___x_969_;
}
case 8:
{
lean_object* v_declName_970_; lean_object* v_type_971_; lean_object* v_value_972_; lean_object* v_body_973_; uint8_t v_nondep_974_; lean_object* v___x_975_; lean_object* v___x_976_; lean_object* v___x_977_; lean_object* v___x_978_; lean_object* v___x_979_; lean_object* v___x_980_; lean_object* v___x_981_; lean_object* v___x_982_; lean_object* v___x_983_; 
v_declName_970_ = lean_ctor_get(v_x_909_, 0);
lean_inc(v_declName_970_);
v_type_971_ = lean_ctor_get(v_x_909_, 1);
lean_inc_ref(v_type_971_);
v_value_972_ = lean_ctor_get(v_x_909_, 2);
lean_inc_ref(v_value_972_);
v_body_973_ = lean_ctor_get(v_x_909_, 3);
lean_inc_ref(v_body_973_);
v_nondep_974_ = lean_ctor_get_uint8(v_x_909_, sizeof(void*)*4 + 8);
lean_dec_ref_known(v_x_909_, 4);
v___x_975_ = lean_obj_once(&lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__31, &lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__31_once, _init_lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__31);
v___x_976_ = l___private_Lean_ToExpr_0__Lean_Name_toExprAux(v_declName_970_);
v___x_977_ = l_Lean_Expr_app___override(v___x_975_, v___x_976_);
v___x_978_ = lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr(v_type_971_);
v___x_979_ = l_Lean_Expr_app___override(v___x_977_, v___x_978_);
v___x_980_ = lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr(v_value_972_);
v___x_981_ = l_Lean_Expr_app___override(v___x_979_, v___x_980_);
v___x_982_ = lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr(v_body_973_);
v___x_983_ = l_Lean_Expr_app___override(v___x_981_, v___x_982_);
if (v_nondep_974_ == 0)
{
lean_object* v___x_984_; lean_object* v___x_985_; 
v___x_984_ = lean_obj_once(&lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr___closed__11, &lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr___closed__11_once, _init_lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr___closed__11);
v___x_985_ = l_Lean_Expr_app___override(v___x_983_, v___x_984_);
return v___x_985_;
}
else
{
lean_object* v___x_986_; lean_object* v___x_987_; 
v___x_986_ = lean_obj_once(&lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr___closed__14, &lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr___closed__14_once, _init_lp_mathlib_Mathlib_instToExprSourceInfo__mathlib_toExpr___closed__14);
v___x_987_ = l_Lean_Expr_app___override(v___x_983_, v___x_986_);
return v___x_987_;
}
}
case 9:
{
lean_object* v_a_988_; lean_object* v___x_989_; 
v_a_988_ = lean_ctor_get(v_x_909_, 0);
lean_inc_ref(v_a_988_);
lean_dec_ref_known(v_x_909_, 1);
v___x_989_ = lean_obj_once(&lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__34, &lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__34_once, _init_lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__34);
if (lean_obj_tag(v_a_988_) == 0)
{
lean_object* v___x_990_; lean_object* v___x_991_; lean_object* v___x_992_; lean_object* v___x_993_; 
v___x_990_ = lean_obj_once(&lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__38, &lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__38_once, _init_lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__38);
v___x_991_ = l_Lean_Expr_lit___override(v_a_988_);
v___x_992_ = l_Lean_Expr_app___override(v___x_990_, v___x_991_);
v___x_993_ = l_Lean_Expr_app___override(v___x_989_, v___x_992_);
return v___x_993_;
}
else
{
lean_object* v___x_994_; lean_object* v___x_995_; lean_object* v___x_996_; lean_object* v___x_997_; 
v___x_994_ = lean_obj_once(&lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__41, &lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__41_once, _init_lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__41);
v___x_995_ = l_Lean_Expr_lit___override(v_a_988_);
v___x_996_ = l_Lean_Expr_app___override(v___x_994_, v___x_995_);
v___x_997_ = l_Lean_Expr_app___override(v___x_989_, v___x_996_);
return v___x_997_;
}
}
case 10:
{
lean_object* v_data_998_; lean_object* v_expr_999_; lean_object* v___x_1000_; lean_object* v___x_1001_; lean_object* v___x_1002_; lean_object* v___x_1003_; lean_object* v___x_1004_; 
v_data_998_ = lean_ctor_get(v_x_909_, 0);
lean_inc(v_data_998_);
v_expr_999_ = lean_ctor_get(v_x_909_, 1);
lean_inc_ref(v_expr_999_);
lean_dec_ref_known(v_x_909_, 2);
v___x_1000_ = lean_obj_once(&lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__44, &lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__44_once, _init_lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__44);
v___x_1001_ = lp_mathlib___private_Mathlib_Tactic_ToExpr_0__Mathlib_toExprMData(v_data_998_);
lean_dec(v_data_998_);
v___x_1002_ = l_Lean_Expr_app___override(v___x_1000_, v___x_1001_);
v___x_1003_ = lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr(v_expr_999_);
v___x_1004_ = l_Lean_Expr_app___override(v___x_1002_, v___x_1003_);
return v___x_1004_;
}
default: 
{
lean_object* v_typeName_1005_; lean_object* v_idx_1006_; lean_object* v_struct_1007_; lean_object* v___x_1008_; lean_object* v___x_1009_; lean_object* v___x_1010_; lean_object* v___x_1011_; lean_object* v___x_1012_; lean_object* v___x_1013_; lean_object* v___x_1014_; 
v_typeName_1005_ = lean_ctor_get(v_x_909_, 0);
lean_inc(v_typeName_1005_);
v_idx_1006_ = lean_ctor_get(v_x_909_, 1);
lean_inc(v_idx_1006_);
v_struct_1007_ = lean_ctor_get(v_x_909_, 2);
lean_inc_ref(v_struct_1007_);
lean_dec_ref_known(v_x_909_, 3);
v___x_1008_ = lean_obj_once(&lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__47, &lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__47_once, _init_lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr___closed__47);
v___x_1009_ = l___private_Lean_ToExpr_0__Lean_Name_toExprAux(v_typeName_1005_);
v___x_1010_ = l_Lean_Expr_app___override(v___x_1008_, v___x_1009_);
v___x_1011_ = l_Lean_mkNatLit(v_idx_1006_);
v___x_1012_ = l_Lean_Expr_app___override(v___x_1010_, v___x_1011_);
v___x_1013_ = lp_mathlib_Mathlib_instToExprExpr__mathlib_toExpr(v_struct_1007_);
v___x_1014_ = l_Lean_Expr_app___override(v___x_1012_, v___x_1013_);
return v___x_1014_;
}
}
}
}
static lean_object* _init_lp_mathlib_Mathlib_instToExprExpr__mathlib___closed__2(void){
_start:
{
lean_object* v___x_1019_; lean_object* v___x_1020_; lean_object* v___x_1021_; 
v___x_1019_ = lean_box(0);
v___x_1020_ = ((lean_object*)(lp_mathlib_Mathlib_instToExprExpr__mathlib___closed__1));
v___x_1021_ = l_Lean_Expr_const___override(v___x_1020_, v___x_1019_);
return v___x_1021_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_instToExprExpr__mathlib___closed__3(void){
_start:
{
lean_object* v___x_1022_; lean_object* v___x_1023_; lean_object* v___x_1024_; 
v___x_1022_ = lean_obj_once(&lp_mathlib_Mathlib_instToExprExpr__mathlib___closed__2, &lp_mathlib_Mathlib_instToExprExpr__mathlib___closed__2_once, _init_lp_mathlib_Mathlib_instToExprExpr__mathlib___closed__2);
v___x_1023_ = ((lean_object*)(lp_mathlib_Mathlib_instToExprExpr__mathlib___closed__0));
v___x_1024_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1024_, 0, v___x_1023_);
lean_ctor_set(v___x_1024_, 1, v___x_1022_);
return v___x_1024_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_instToExprExpr__mathlib(void){
_start:
{
lean_object* v___x_1025_; 
v___x_1025_ = lean_obj_once(&lp_mathlib_Mathlib_instToExprExpr__mathlib___closed__3, &lp_mathlib_Mathlib_instToExprExpr__mathlib___closed__3_once, _init_lp_mathlib_Mathlib_instToExprExpr__mathlib___closed__3);
return v___x_1025_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Init(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Tactic_ToExpr(uint8_t builtin) {
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
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Tactic_ToExpr(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Mathlib_instToExprRaw__mathlib = _init_lp_mathlib_Mathlib_instToExprRaw__mathlib();
lean_mark_persistent(lp_mathlib_Mathlib_instToExprRaw__mathlib);
lp_mathlib_Mathlib_instToExprRaw__mathlib__1 = _init_lp_mathlib_Mathlib_instToExprRaw__mathlib__1();
lean_mark_persistent(lp_mathlib_Mathlib_instToExprRaw__mathlib__1);
lp_mathlib_Mathlib_instToExprSourceInfo__mathlib = _init_lp_mathlib_Mathlib_instToExprSourceInfo__mathlib();
lean_mark_persistent(lp_mathlib_Mathlib_instToExprSourceInfo__mathlib);
lp_mathlib_Mathlib_instToExprSyntax__mathlib = _init_lp_mathlib_Mathlib_instToExprSyntax__mathlib();
lean_mark_persistent(lp_mathlib_Mathlib_instToExprSyntax__mathlib);
lp_mathlib_Mathlib_instToExprMData__mathlib = _init_lp_mathlib_Mathlib_instToExprMData__mathlib();
lean_mark_persistent(lp_mathlib_Mathlib_instToExprMData__mathlib);
lp_mathlib_Mathlib_instToExprMVarId__mathlib = _init_lp_mathlib_Mathlib_instToExprMVarId__mathlib();
lean_mark_persistent(lp_mathlib_Mathlib_instToExprMVarId__mathlib);
lp_mathlib_Mathlib_instToExprLevelMVarId__mathlib = _init_lp_mathlib_Mathlib_instToExprLevelMVarId__mathlib();
lean_mark_persistent(lp_mathlib_Mathlib_instToExprLevelMVarId__mathlib);
lp_mathlib_Mathlib_instToExprLevel__mathlib = _init_lp_mathlib_Mathlib_instToExprLevel__mathlib();
lean_mark_persistent(lp_mathlib_Mathlib_instToExprLevel__mathlib);
lp_mathlib_Mathlib_instToExprBinderInfo__mathlib = _init_lp_mathlib_Mathlib_instToExprBinderInfo__mathlib();
lean_mark_persistent(lp_mathlib_Mathlib_instToExprBinderInfo__mathlib);
lp_mathlib_Mathlib_instToExprExpr__mathlib = _init_lp_mathlib_Mathlib_instToExprExpr__mathlib();
lean_mark_persistent(lp_mathlib_Mathlib_instToExprExpr__mathlib);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Init(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_ToExpr(uint8_t builtin) {
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
res = runtime_initialize_mathlib_Mathlib_Tactic_ToExpr(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Tactic_ToExpr(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Tactic_ToExpr(builtin);
}
#ifdef __cplusplus
}
#endif
