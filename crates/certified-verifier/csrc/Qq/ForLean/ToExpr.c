// Lean compiler output
// Module: Qq.ForLean.ToExpr
// Imports: public import Init public meta import Init public import Lean.ToExpr
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
lean_object* l_Lean_Name_mkStr3(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* l_Lean_Expr_const___override(lean_object*, lean_object*);
lean_object* l_Lean_Expr_app___override(lean_object*, lean_object*);
lean_object* l_Lean_mkAppB(lean_object*, lean_object*, lean_object*);
lean_object* l___private_Lean_ToExpr_0__Lean_Name_toExprAux(lean_object*);
lean_object* l_Id_instMonad___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__6(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__3(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__2___boxed(lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_mkStrLit(lean_object*);
lean_object* l_Lean_mkApp3(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_mkConst(lean_object*, lean_object*);
lean_object* l_Lean_mkNatLit(lean_object*);
lean_object* lean_nat_to_int(lean_object*);
uint8_t lean_int_dec_le(lean_object*, lean_object*);
lean_object* l_Lean_Level_ofNat(lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* lean_int_neg(lean_object*);
lean_object* l_Int_toNat(lean_object*);
lean_object* l_Lean_instToExprInt_mkNat(lean_object*);
lean_object* l_List_forIn_x27_loop___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_mkApp4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_mkApp5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_lit___override(lean_object*);
static const lean_string_object lp_Qq_instToExprMVarId__qq___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_Qq_instToExprMVarId__qq___lam__0___closed__0 = (const lean_object*)&lp_Qq_instToExprMVarId__qq___lam__0___closed__0_value;
static const lean_string_object lp_Qq_instToExprMVarId__qq___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "MVarId"};
static const lean_object* lp_Qq_instToExprMVarId__qq___lam__0___closed__1 = (const lean_object*)&lp_Qq_instToExprMVarId__qq___lam__0___closed__1_value;
static const lean_string_object lp_Qq_instToExprMVarId__qq___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "mk"};
static const lean_object* lp_Qq_instToExprMVarId__qq___lam__0___closed__2 = (const lean_object*)&lp_Qq_instToExprMVarId__qq___lam__0___closed__2_value;
static const lean_ctor_object lp_Qq_instToExprMVarId__qq___lam__0___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_Qq_instToExprMVarId__qq___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_Qq_instToExprMVarId__qq___lam__0___closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_instToExprMVarId__qq___lam__0___closed__3_value_aux_0),((lean_object*)&lp_Qq_instToExprMVarId__qq___lam__0___closed__1_value),LEAN_SCALAR_PTR_LITERAL(177, 186, 234, 138, 172, 166, 87, 74)}};
static const lean_ctor_object lp_Qq_instToExprMVarId__qq___lam__0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_instToExprMVarId__qq___lam__0___closed__3_value_aux_1),((lean_object*)&lp_Qq_instToExprMVarId__qq___lam__0___closed__2_value),LEAN_SCALAR_PTR_LITERAL(93, 44, 60, 136, 72, 250, 230, 141)}};
static const lean_object* lp_Qq_instToExprMVarId__qq___lam__0___closed__3 = (const lean_object*)&lp_Qq_instToExprMVarId__qq___lam__0___closed__3_value;
static lean_once_cell_t lp_Qq_instToExprMVarId__qq___lam__0___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Qq_instToExprMVarId__qq___lam__0___closed__4;
LEAN_EXPORT lean_object* lp_Qq_instToExprMVarId__qq___lam__0(lean_object*);
static const lean_closure_object lp_Qq_instToExprMVarId__qq___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_Qq_instToExprMVarId__qq___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_Qq_instToExprMVarId__qq___closed__0 = (const lean_object*)&lp_Qq_instToExprMVarId__qq___closed__0_value;
static const lean_ctor_object lp_Qq_instToExprMVarId__qq___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_Qq_instToExprMVarId__qq___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_Qq_instToExprMVarId__qq___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_instToExprMVarId__qq___closed__1_value_aux_0),((lean_object*)&lp_Qq_instToExprMVarId__qq___lam__0___closed__1_value),LEAN_SCALAR_PTR_LITERAL(177, 186, 234, 138, 172, 166, 87, 74)}};
static const lean_object* lp_Qq_instToExprMVarId__qq___closed__1 = (const lean_object*)&lp_Qq_instToExprMVarId__qq___closed__1_value;
static lean_once_cell_t lp_Qq_instToExprMVarId__qq___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Qq_instToExprMVarId__qq___closed__2;
static lean_once_cell_t lp_Qq_instToExprMVarId__qq___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Qq_instToExprMVarId__qq___closed__3;
LEAN_EXPORT lean_object* lp_Qq_instToExprMVarId__qq;
static const lean_string_object lp_Qq_instToExprLevelMVarId__qq___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "LevelMVarId"};
static const lean_object* lp_Qq_instToExprLevelMVarId__qq___lam__0___closed__0 = (const lean_object*)&lp_Qq_instToExprLevelMVarId__qq___lam__0___closed__0_value;
static const lean_ctor_object lp_Qq_instToExprLevelMVarId__qq___lam__0___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_Qq_instToExprMVarId__qq___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_Qq_instToExprLevelMVarId__qq___lam__0___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_instToExprLevelMVarId__qq___lam__0___closed__1_value_aux_0),((lean_object*)&lp_Qq_instToExprLevelMVarId__qq___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(89, 60, 85, 89, 175, 240, 129, 147)}};
static const lean_ctor_object lp_Qq_instToExprLevelMVarId__qq___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_instToExprLevelMVarId__qq___lam__0___closed__1_value_aux_1),((lean_object*)&lp_Qq_instToExprMVarId__qq___lam__0___closed__2_value),LEAN_SCALAR_PTR_LITERAL(213, 157, 226, 48, 182, 72, 20, 234)}};
static const lean_object* lp_Qq_instToExprLevelMVarId__qq___lam__0___closed__1 = (const lean_object*)&lp_Qq_instToExprLevelMVarId__qq___lam__0___closed__1_value;
static lean_once_cell_t lp_Qq_instToExprLevelMVarId__qq___lam__0___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Qq_instToExprLevelMVarId__qq___lam__0___closed__2;
LEAN_EXPORT lean_object* lp_Qq_instToExprLevelMVarId__qq___lam__0(lean_object*);
static const lean_closure_object lp_Qq_instToExprLevelMVarId__qq___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_Qq_instToExprLevelMVarId__qq___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_Qq_instToExprLevelMVarId__qq___closed__0 = (const lean_object*)&lp_Qq_instToExprLevelMVarId__qq___closed__0_value;
static const lean_ctor_object lp_Qq_instToExprLevelMVarId__qq___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_Qq_instToExprMVarId__qq___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_Qq_instToExprLevelMVarId__qq___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_instToExprLevelMVarId__qq___closed__1_value_aux_0),((lean_object*)&lp_Qq_instToExprLevelMVarId__qq___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(89, 60, 85, 89, 175, 240, 129, 147)}};
static const lean_object* lp_Qq_instToExprLevelMVarId__qq___closed__1 = (const lean_object*)&lp_Qq_instToExprLevelMVarId__qq___closed__1_value;
static lean_once_cell_t lp_Qq_instToExprLevelMVarId__qq___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Qq_instToExprLevelMVarId__qq___closed__2;
static lean_once_cell_t lp_Qq_instToExprLevelMVarId__qq___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Qq_instToExprLevelMVarId__qq___closed__3;
LEAN_EXPORT lean_object* lp_Qq_instToExprLevelMVarId__qq;
static const lean_string_object lp_Qq_toExprLevel___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "Level"};
static const lean_object* lp_Qq_toExprLevel___closed__0 = (const lean_object*)&lp_Qq_toExprLevel___closed__0_value;
static const lean_string_object lp_Qq_toExprLevel___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "zero"};
static const lean_object* lp_Qq_toExprLevel___closed__1 = (const lean_object*)&lp_Qq_toExprLevel___closed__1_value;
static const lean_ctor_object lp_Qq_toExprLevel___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_Qq_instToExprMVarId__qq___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_Qq_toExprLevel___closed__2_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_toExprLevel___closed__2_value_aux_0),((lean_object*)&lp_Qq_toExprLevel___closed__0_value),LEAN_SCALAR_PTR_LITERAL(165, 140, 25, 163, 179, 32, 245, 95)}};
static const lean_ctor_object lp_Qq_toExprLevel___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_toExprLevel___closed__2_value_aux_1),((lean_object*)&lp_Qq_toExprLevel___closed__1_value),LEAN_SCALAR_PTR_LITERAL(13, 6, 49, 141, 220, 30, 84, 149)}};
static const lean_object* lp_Qq_toExprLevel___closed__2 = (const lean_object*)&lp_Qq_toExprLevel___closed__2_value;
static lean_once_cell_t lp_Qq_toExprLevel___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Qq_toExprLevel___closed__3;
static const lean_string_object lp_Qq_toExprLevel___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "succ"};
static const lean_object* lp_Qq_toExprLevel___closed__4 = (const lean_object*)&lp_Qq_toExprLevel___closed__4_value;
static const lean_ctor_object lp_Qq_toExprLevel___closed__5_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_Qq_instToExprMVarId__qq___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_Qq_toExprLevel___closed__5_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_toExprLevel___closed__5_value_aux_0),((lean_object*)&lp_Qq_toExprLevel___closed__0_value),LEAN_SCALAR_PTR_LITERAL(165, 140, 25, 163, 179, 32, 245, 95)}};
static const lean_ctor_object lp_Qq_toExprLevel___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_toExprLevel___closed__5_value_aux_1),((lean_object*)&lp_Qq_toExprLevel___closed__4_value),LEAN_SCALAR_PTR_LITERAL(227, 93, 133, 102, 36, 205, 79, 205)}};
static const lean_object* lp_Qq_toExprLevel___closed__5 = (const lean_object*)&lp_Qq_toExprLevel___closed__5_value;
static lean_once_cell_t lp_Qq_toExprLevel___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Qq_toExprLevel___closed__6;
static const lean_string_object lp_Qq_toExprLevel___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "max"};
static const lean_object* lp_Qq_toExprLevel___closed__7 = (const lean_object*)&lp_Qq_toExprLevel___closed__7_value;
static const lean_ctor_object lp_Qq_toExprLevel___closed__8_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_Qq_instToExprMVarId__qq___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_Qq_toExprLevel___closed__8_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_toExprLevel___closed__8_value_aux_0),((lean_object*)&lp_Qq_toExprLevel___closed__0_value),LEAN_SCALAR_PTR_LITERAL(165, 140, 25, 163, 179, 32, 245, 95)}};
static const lean_ctor_object lp_Qq_toExprLevel___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_toExprLevel___closed__8_value_aux_1),((lean_object*)&lp_Qq_toExprLevel___closed__7_value),LEAN_SCALAR_PTR_LITERAL(163, 196, 232, 122, 251, 166, 170, 227)}};
static const lean_object* lp_Qq_toExprLevel___closed__8 = (const lean_object*)&lp_Qq_toExprLevel___closed__8_value;
static lean_once_cell_t lp_Qq_toExprLevel___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Qq_toExprLevel___closed__9;
static const lean_string_object lp_Qq_toExprLevel___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "imax"};
static const lean_object* lp_Qq_toExprLevel___closed__10 = (const lean_object*)&lp_Qq_toExprLevel___closed__10_value;
static const lean_ctor_object lp_Qq_toExprLevel___closed__11_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_Qq_instToExprMVarId__qq___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_Qq_toExprLevel___closed__11_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_toExprLevel___closed__11_value_aux_0),((lean_object*)&lp_Qq_toExprLevel___closed__0_value),LEAN_SCALAR_PTR_LITERAL(165, 140, 25, 163, 179, 32, 245, 95)}};
static const lean_ctor_object lp_Qq_toExprLevel___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_toExprLevel___closed__11_value_aux_1),((lean_object*)&lp_Qq_toExprLevel___closed__10_value),LEAN_SCALAR_PTR_LITERAL(13, 164, 87, 20, 224, 129, 213, 91)}};
static const lean_object* lp_Qq_toExprLevel___closed__11 = (const lean_object*)&lp_Qq_toExprLevel___closed__11_value;
static lean_once_cell_t lp_Qq_toExprLevel___closed__12_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Qq_toExprLevel___closed__12;
static const lean_string_object lp_Qq_toExprLevel___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "param"};
static const lean_object* lp_Qq_toExprLevel___closed__13 = (const lean_object*)&lp_Qq_toExprLevel___closed__13_value;
static const lean_ctor_object lp_Qq_toExprLevel___closed__14_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_Qq_instToExprMVarId__qq___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_Qq_toExprLevel___closed__14_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_toExprLevel___closed__14_value_aux_0),((lean_object*)&lp_Qq_toExprLevel___closed__0_value),LEAN_SCALAR_PTR_LITERAL(165, 140, 25, 163, 179, 32, 245, 95)}};
static const lean_ctor_object lp_Qq_toExprLevel___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_toExprLevel___closed__14_value_aux_1),((lean_object*)&lp_Qq_toExprLevel___closed__13_value),LEAN_SCALAR_PTR_LITERAL(196, 134, 94, 195, 247, 235, 245, 84)}};
static const lean_object* lp_Qq_toExprLevel___closed__14 = (const lean_object*)&lp_Qq_toExprLevel___closed__14_value;
static lean_once_cell_t lp_Qq_toExprLevel___closed__15_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Qq_toExprLevel___closed__15;
static const lean_string_object lp_Qq_toExprLevel___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "mvar"};
static const lean_object* lp_Qq_toExprLevel___closed__16 = (const lean_object*)&lp_Qq_toExprLevel___closed__16_value;
static const lean_ctor_object lp_Qq_toExprLevel___closed__17_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_Qq_instToExprMVarId__qq___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_Qq_toExprLevel___closed__17_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_toExprLevel___closed__17_value_aux_0),((lean_object*)&lp_Qq_toExprLevel___closed__0_value),LEAN_SCALAR_PTR_LITERAL(165, 140, 25, 163, 179, 32, 245, 95)}};
static const lean_ctor_object lp_Qq_toExprLevel___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_toExprLevel___closed__17_value_aux_1),((lean_object*)&lp_Qq_toExprLevel___closed__16_value),LEAN_SCALAR_PTR_LITERAL(33, 188, 104, 40, 236, 34, 24, 77)}};
static const lean_object* lp_Qq_toExprLevel___closed__17 = (const lean_object*)&lp_Qq_toExprLevel___closed__17_value;
static lean_once_cell_t lp_Qq_toExprLevel___closed__18_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Qq_toExprLevel___closed__18;
LEAN_EXPORT lean_object* lp_Qq_toExprLevel(lean_object*);
static const lean_closure_object lp_Qq_instToExprLevel__qq___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_Qq_toExprLevel, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_Qq_instToExprLevel__qq___closed__0 = (const lean_object*)&lp_Qq_instToExprLevel__qq___closed__0_value;
static const lean_ctor_object lp_Qq_instToExprLevel__qq___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_Qq_instToExprMVarId__qq___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_Qq_instToExprLevel__qq___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_instToExprLevel__qq___closed__1_value_aux_0),((lean_object*)&lp_Qq_toExprLevel___closed__0_value),LEAN_SCALAR_PTR_LITERAL(165, 140, 25, 163, 179, 32, 245, 95)}};
static const lean_object* lp_Qq_instToExprLevel__qq___closed__1 = (const lean_object*)&lp_Qq_instToExprLevel__qq___closed__1_value;
static lean_once_cell_t lp_Qq_instToExprLevel__qq___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Qq_instToExprLevel__qq___closed__2;
static lean_once_cell_t lp_Qq_instToExprLevel__qq___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Qq_instToExprLevel__qq___closed__3;
LEAN_EXPORT lean_object* lp_Qq_instToExprLevel__qq;
static const lean_string_object lp_Qq_instToExprBinderInfo__qq___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "BinderInfo"};
static const lean_object* lp_Qq_instToExprBinderInfo__qq___lam__0___closed__0 = (const lean_object*)&lp_Qq_instToExprBinderInfo__qq___lam__0___closed__0_value;
static const lean_string_object lp_Qq_instToExprBinderInfo__qq___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "default"};
static const lean_object* lp_Qq_instToExprBinderInfo__qq___lam__0___closed__1 = (const lean_object*)&lp_Qq_instToExprBinderInfo__qq___lam__0___closed__1_value;
static const lean_ctor_object lp_Qq_instToExprBinderInfo__qq___lam__0___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_Qq_instToExprMVarId__qq___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_Qq_instToExprBinderInfo__qq___lam__0___closed__2_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_instToExprBinderInfo__qq___lam__0___closed__2_value_aux_0),((lean_object*)&lp_Qq_instToExprBinderInfo__qq___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(25, 16, 226, 245, 106, 161, 19, 213)}};
static const lean_ctor_object lp_Qq_instToExprBinderInfo__qq___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_instToExprBinderInfo__qq___lam__0___closed__2_value_aux_1),((lean_object*)&lp_Qq_instToExprBinderInfo__qq___lam__0___closed__1_value),LEAN_SCALAR_PTR_LITERAL(39, 15, 252, 127, 213, 76, 105, 203)}};
static const lean_object* lp_Qq_instToExprBinderInfo__qq___lam__0___closed__2 = (const lean_object*)&lp_Qq_instToExprBinderInfo__qq___lam__0___closed__2_value;
static lean_once_cell_t lp_Qq_instToExprBinderInfo__qq___lam__0___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Qq_instToExprBinderInfo__qq___lam__0___closed__3;
static const lean_string_object lp_Qq_instToExprBinderInfo__qq___lam__0___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "implicit"};
static const lean_object* lp_Qq_instToExprBinderInfo__qq___lam__0___closed__4 = (const lean_object*)&lp_Qq_instToExprBinderInfo__qq___lam__0___closed__4_value;
static const lean_ctor_object lp_Qq_instToExprBinderInfo__qq___lam__0___closed__5_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_Qq_instToExprMVarId__qq___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_Qq_instToExprBinderInfo__qq___lam__0___closed__5_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_instToExprBinderInfo__qq___lam__0___closed__5_value_aux_0),((lean_object*)&lp_Qq_instToExprBinderInfo__qq___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(25, 16, 226, 245, 106, 161, 19, 213)}};
static const lean_ctor_object lp_Qq_instToExprBinderInfo__qq___lam__0___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_instToExprBinderInfo__qq___lam__0___closed__5_value_aux_1),((lean_object*)&lp_Qq_instToExprBinderInfo__qq___lam__0___closed__4_value),LEAN_SCALAR_PTR_LITERAL(251, 202, 67, 228, 64, 219, 133, 236)}};
static const lean_object* lp_Qq_instToExprBinderInfo__qq___lam__0___closed__5 = (const lean_object*)&lp_Qq_instToExprBinderInfo__qq___lam__0___closed__5_value;
static lean_once_cell_t lp_Qq_instToExprBinderInfo__qq___lam__0___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Qq_instToExprBinderInfo__qq___lam__0___closed__6;
static const lean_string_object lp_Qq_instToExprBinderInfo__qq___lam__0___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "strictImplicit"};
static const lean_object* lp_Qq_instToExprBinderInfo__qq___lam__0___closed__7 = (const lean_object*)&lp_Qq_instToExprBinderInfo__qq___lam__0___closed__7_value;
static const lean_ctor_object lp_Qq_instToExprBinderInfo__qq___lam__0___closed__8_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_Qq_instToExprMVarId__qq___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_Qq_instToExprBinderInfo__qq___lam__0___closed__8_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_instToExprBinderInfo__qq___lam__0___closed__8_value_aux_0),((lean_object*)&lp_Qq_instToExprBinderInfo__qq___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(25, 16, 226, 245, 106, 161, 19, 213)}};
static const lean_ctor_object lp_Qq_instToExprBinderInfo__qq___lam__0___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_instToExprBinderInfo__qq___lam__0___closed__8_value_aux_1),((lean_object*)&lp_Qq_instToExprBinderInfo__qq___lam__0___closed__7_value),LEAN_SCALAR_PTR_LITERAL(210, 36, 185, 75, 137, 139, 69, 221)}};
static const lean_object* lp_Qq_instToExprBinderInfo__qq___lam__0___closed__8 = (const lean_object*)&lp_Qq_instToExprBinderInfo__qq___lam__0___closed__8_value;
static lean_once_cell_t lp_Qq_instToExprBinderInfo__qq___lam__0___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Qq_instToExprBinderInfo__qq___lam__0___closed__9;
static const lean_string_object lp_Qq_instToExprBinderInfo__qq___lam__0___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "instImplicit"};
static const lean_object* lp_Qq_instToExprBinderInfo__qq___lam__0___closed__10 = (const lean_object*)&lp_Qq_instToExprBinderInfo__qq___lam__0___closed__10_value;
static const lean_ctor_object lp_Qq_instToExprBinderInfo__qq___lam__0___closed__11_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_Qq_instToExprMVarId__qq___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_Qq_instToExprBinderInfo__qq___lam__0___closed__11_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_instToExprBinderInfo__qq___lam__0___closed__11_value_aux_0),((lean_object*)&lp_Qq_instToExprBinderInfo__qq___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(25, 16, 226, 245, 106, 161, 19, 213)}};
static const lean_ctor_object lp_Qq_instToExprBinderInfo__qq___lam__0___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_instToExprBinderInfo__qq___lam__0___closed__11_value_aux_1),((lean_object*)&lp_Qq_instToExprBinderInfo__qq___lam__0___closed__10_value),LEAN_SCALAR_PTR_LITERAL(121, 65, 31, 57, 146, 51, 125, 181)}};
static const lean_object* lp_Qq_instToExprBinderInfo__qq___lam__0___closed__11 = (const lean_object*)&lp_Qq_instToExprBinderInfo__qq___lam__0___closed__11_value;
static lean_once_cell_t lp_Qq_instToExprBinderInfo__qq___lam__0___closed__12_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Qq_instToExprBinderInfo__qq___lam__0___closed__12;
LEAN_EXPORT lean_object* lp_Qq_instToExprBinderInfo__qq___lam__0(uint8_t);
LEAN_EXPORT lean_object* lp_Qq_instToExprBinderInfo__qq___lam__0___boxed(lean_object*);
static const lean_closure_object lp_Qq_instToExprBinderInfo__qq___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_Qq_instToExprBinderInfo__qq___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_Qq_instToExprBinderInfo__qq___closed__0 = (const lean_object*)&lp_Qq_instToExprBinderInfo__qq___closed__0_value;
static const lean_ctor_object lp_Qq_instToExprBinderInfo__qq___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_Qq_instToExprMVarId__qq___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_Qq_instToExprBinderInfo__qq___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_instToExprBinderInfo__qq___closed__1_value_aux_0),((lean_object*)&lp_Qq_instToExprBinderInfo__qq___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(25, 16, 226, 245, 106, 161, 19, 213)}};
static const lean_object* lp_Qq_instToExprBinderInfo__qq___closed__1 = (const lean_object*)&lp_Qq_instToExprBinderInfo__qq___closed__1_value;
static lean_once_cell_t lp_Qq_instToExprBinderInfo__qq___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Qq_instToExprBinderInfo__qq___closed__2;
static lean_once_cell_t lp_Qq_instToExprBinderInfo__qq___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Qq_instToExprBinderInfo__qq___closed__3;
LEAN_EXPORT lean_object* lp_Qq_instToExprBinderInfo__qq;
static const lean_string_object lp_Qq_instToExprMData__qq___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "KVMap"};
static const lean_object* lp_Qq_instToExprMData__qq___lam__0___closed__0 = (const lean_object*)&lp_Qq_instToExprMData__qq___lam__0___closed__0_value;
static const lean_string_object lp_Qq_instToExprMData__qq___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "setString"};
static const lean_object* lp_Qq_instToExprMData__qq___lam__0___closed__1 = (const lean_object*)&lp_Qq_instToExprMData__qq___lam__0___closed__1_value;
static const lean_string_object lp_Qq_instToExprMData__qq___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "setBool"};
static const lean_object* lp_Qq_instToExprMData__qq___lam__0___closed__2 = (const lean_object*)&lp_Qq_instToExprMData__qq___lam__0___closed__2_value;
static const lean_string_object lp_Qq_instToExprMData__qq___lam__0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Bool"};
static const lean_object* lp_Qq_instToExprMData__qq___lam__0___closed__3 = (const lean_object*)&lp_Qq_instToExprMData__qq___lam__0___closed__3_value;
static const lean_string_object lp_Qq_instToExprMData__qq___lam__0___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "false"};
static const lean_object* lp_Qq_instToExprMData__qq___lam__0___closed__4 = (const lean_object*)&lp_Qq_instToExprMData__qq___lam__0___closed__4_value;
static const lean_ctor_object lp_Qq_instToExprMData__qq___lam__0___closed__5_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_Qq_instToExprMData__qq___lam__0___closed__3_value),LEAN_SCALAR_PTR_LITERAL(250, 44, 198, 216, 184, 195, 199, 178)}};
static const lean_ctor_object lp_Qq_instToExprMData__qq___lam__0___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_instToExprMData__qq___lam__0___closed__5_value_aux_0),((lean_object*)&lp_Qq_instToExprMData__qq___lam__0___closed__4_value),LEAN_SCALAR_PTR_LITERAL(117, 151, 161, 190, 111, 237, 188, 218)}};
static const lean_object* lp_Qq_instToExprMData__qq___lam__0___closed__5 = (const lean_object*)&lp_Qq_instToExprMData__qq___lam__0___closed__5_value;
static const lean_string_object lp_Qq_instToExprMData__qq___lam__0___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "true"};
static const lean_object* lp_Qq_instToExprMData__qq___lam__0___closed__6 = (const lean_object*)&lp_Qq_instToExprMData__qq___lam__0___closed__6_value;
static const lean_ctor_object lp_Qq_instToExprMData__qq___lam__0___closed__7_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_Qq_instToExprMData__qq___lam__0___closed__3_value),LEAN_SCALAR_PTR_LITERAL(250, 44, 198, 216, 184, 195, 199, 178)}};
static const lean_ctor_object lp_Qq_instToExprMData__qq___lam__0___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_instToExprMData__qq___lam__0___closed__7_value_aux_0),((lean_object*)&lp_Qq_instToExprMData__qq___lam__0___closed__6_value),LEAN_SCALAR_PTR_LITERAL(22, 245, 194, 28, 184, 9, 113, 128)}};
static const lean_object* lp_Qq_instToExprMData__qq___lam__0___closed__7 = (const lean_object*)&lp_Qq_instToExprMData__qq___lam__0___closed__7_value;
static const lean_string_object lp_Qq_instToExprMData__qq___lam__0___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "setName"};
static const lean_object* lp_Qq_instToExprMData__qq___lam__0___closed__8 = (const lean_object*)&lp_Qq_instToExprMData__qq___lam__0___closed__8_value;
static const lean_string_object lp_Qq_instToExprMData__qq___lam__0___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "setNat"};
static const lean_object* lp_Qq_instToExprMData__qq___lam__0___closed__9 = (const lean_object*)&lp_Qq_instToExprMData__qq___lam__0___closed__9_value;
static const lean_string_object lp_Qq_instToExprMData__qq___lam__0___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "setInt"};
static const lean_object* lp_Qq_instToExprMData__qq___lam__0___closed__10 = (const lean_object*)&lp_Qq_instToExprMData__qq___lam__0___closed__10_value;
static lean_once_cell_t lp_Qq_instToExprMData__qq___lam__0___closed__11_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Qq_instToExprMData__qq___lam__0___closed__11;
static const lean_string_object lp_Qq_instToExprMData__qq___lam__0___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Neg"};
static const lean_object* lp_Qq_instToExprMData__qq___lam__0___closed__12 = (const lean_object*)&lp_Qq_instToExprMData__qq___lam__0___closed__12_value;
static const lean_string_object lp_Qq_instToExprMData__qq___lam__0___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "neg"};
static const lean_object* lp_Qq_instToExprMData__qq___lam__0___closed__13 = (const lean_object*)&lp_Qq_instToExprMData__qq___lam__0___closed__13_value;
static const lean_ctor_object lp_Qq_instToExprMData__qq___lam__0___closed__14_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_Qq_instToExprMData__qq___lam__0___closed__12_value),LEAN_SCALAR_PTR_LITERAL(94, 4, 109, 108, 64, 81, 153, 133)}};
static const lean_ctor_object lp_Qq_instToExprMData__qq___lam__0___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_instToExprMData__qq___lam__0___closed__14_value_aux_0),((lean_object*)&lp_Qq_instToExprMData__qq___lam__0___closed__13_value),LEAN_SCALAR_PTR_LITERAL(105, 26, 70, 221, 245, 238, 127, 238)}};
static const lean_object* lp_Qq_instToExprMData__qq___lam__0___closed__14 = (const lean_object*)&lp_Qq_instToExprMData__qq___lam__0___closed__14_value;
static lean_once_cell_t lp_Qq_instToExprMData__qq___lam__0___closed__15_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Qq_instToExprMData__qq___lam__0___closed__15;
static const lean_string_object lp_Qq_instToExprMData__qq___lam__0___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Int"};
static const lean_object* lp_Qq_instToExprMData__qq___lam__0___closed__16 = (const lean_object*)&lp_Qq_instToExprMData__qq___lam__0___closed__16_value;
static const lean_ctor_object lp_Qq_instToExprMData__qq___lam__0___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_Qq_instToExprMData__qq___lam__0___closed__16_value),LEAN_SCALAR_PTR_LITERAL(61, 25, 98, 154, 117, 127, 69, 97)}};
static const lean_object* lp_Qq_instToExprMData__qq___lam__0___closed__17 = (const lean_object*)&lp_Qq_instToExprMData__qq___lam__0___closed__17_value;
static const lean_string_object lp_Qq_instToExprMData__qq___lam__0___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "instNegInt"};
static const lean_object* lp_Qq_instToExprMData__qq___lam__0___closed__18 = (const lean_object*)&lp_Qq_instToExprMData__qq___lam__0___closed__18_value;
static const lean_ctor_object lp_Qq_instToExprMData__qq___lam__0___closed__19_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_Qq_instToExprMData__qq___lam__0___closed__16_value),LEAN_SCALAR_PTR_LITERAL(61, 25, 98, 154, 117, 127, 69, 97)}};
static const lean_ctor_object lp_Qq_instToExprMData__qq___lam__0___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_instToExprMData__qq___lam__0___closed__19_value_aux_0),((lean_object*)&lp_Qq_instToExprMData__qq___lam__0___closed__18_value),LEAN_SCALAR_PTR_LITERAL(217, 109, 233, 1, 211, 122, 77, 88)}};
static const lean_object* lp_Qq_instToExprMData__qq___lam__0___closed__19 = (const lean_object*)&lp_Qq_instToExprMData__qq___lam__0___closed__19_value;
LEAN_EXPORT lean_object* lp_Qq_instToExprMData__qq___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_Qq_instToExprMData__qq___lam__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "MData"};
static const lean_object* lp_Qq_instToExprMData__qq___lam__1___closed__0 = (const lean_object*)&lp_Qq_instToExprMData__qq___lam__1___closed__0_value;
static const lean_string_object lp_Qq_instToExprMData__qq___lam__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "empty"};
static const lean_object* lp_Qq_instToExprMData__qq___lam__1___closed__1 = (const lean_object*)&lp_Qq_instToExprMData__qq___lam__1___closed__1_value;
static const lean_ctor_object lp_Qq_instToExprMData__qq___lam__1___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_Qq_instToExprMVarId__qq___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_Qq_instToExprMData__qq___lam__1___closed__2_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_instToExprMData__qq___lam__1___closed__2_value_aux_0),((lean_object*)&lp_Qq_instToExprMData__qq___lam__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(146, 59, 178, 55, 17, 203, 226, 154)}};
static const lean_ctor_object lp_Qq_instToExprMData__qq___lam__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_instToExprMData__qq___lam__1___closed__2_value_aux_1),((lean_object*)&lp_Qq_instToExprMData__qq___lam__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(202, 189, 63, 250, 0, 123, 75, 60)}};
static const lean_object* lp_Qq_instToExprMData__qq___lam__1___closed__2 = (const lean_object*)&lp_Qq_instToExprMData__qq___lam__1___closed__2_value;
static const lean_closure_object lp_Qq_instToExprMData__qq___lam__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_Qq_instToExprMData__qq___lam__0, .m_arity = 5, .m_num_fixed = 2, .m_objs = {((lean_object*)&lp_Qq_instToExprMVarId__qq___lam__0___closed__0_value),((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_Qq_instToExprMData__qq___lam__1___closed__3 = (const lean_object*)&lp_Qq_instToExprMData__qq___lam__1___closed__3_value;
static lean_once_cell_t lp_Qq_instToExprMData__qq___lam__1___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Qq_instToExprMData__qq___lam__1___closed__4;
LEAN_EXPORT lean_object* lp_Qq_instToExprMData__qq___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_instToExprMData__qq___lam__1___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_Qq_instToExprMData__qq___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__0, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_Qq_instToExprMData__qq___closed__0 = (const lean_object*)&lp_Qq_instToExprMData__qq___closed__0_value;
static const lean_closure_object lp_Qq_instToExprMData__qq___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__1___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_Qq_instToExprMData__qq___closed__1 = (const lean_object*)&lp_Qq_instToExprMData__qq___closed__1_value;
static const lean_closure_object lp_Qq_instToExprMData__qq___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__2___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_Qq_instToExprMData__qq___closed__2 = (const lean_object*)&lp_Qq_instToExprMData__qq___closed__2_value;
static const lean_closure_object lp_Qq_instToExprMData__qq___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__3, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_Qq_instToExprMData__qq___closed__3 = (const lean_object*)&lp_Qq_instToExprMData__qq___closed__3_value;
static const lean_closure_object lp_Qq_instToExprMData__qq___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__4___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_Qq_instToExprMData__qq___closed__4 = (const lean_object*)&lp_Qq_instToExprMData__qq___closed__4_value;
static const lean_closure_object lp_Qq_instToExprMData__qq___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__5___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_Qq_instToExprMData__qq___closed__5 = (const lean_object*)&lp_Qq_instToExprMData__qq___closed__5_value;
static const lean_closure_object lp_Qq_instToExprMData__qq___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__6, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_Qq_instToExprMData__qq___closed__6 = (const lean_object*)&lp_Qq_instToExprMData__qq___closed__6_value;
static const lean_ctor_object lp_Qq_instToExprMData__qq___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_Qq_instToExprMData__qq___closed__0_value),((lean_object*)&lp_Qq_instToExprMData__qq___closed__1_value)}};
static const lean_object* lp_Qq_instToExprMData__qq___closed__7 = (const lean_object*)&lp_Qq_instToExprMData__qq___closed__7_value;
static const lean_ctor_object lp_Qq_instToExprMData__qq___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*5 + 0, .m_other = 5, .m_tag = 0}, .m_objs = {((lean_object*)&lp_Qq_instToExprMData__qq___closed__7_value),((lean_object*)&lp_Qq_instToExprMData__qq___closed__2_value),((lean_object*)&lp_Qq_instToExprMData__qq___closed__3_value),((lean_object*)&lp_Qq_instToExprMData__qq___closed__4_value),((lean_object*)&lp_Qq_instToExprMData__qq___closed__5_value)}};
static const lean_object* lp_Qq_instToExprMData__qq___closed__8 = (const lean_object*)&lp_Qq_instToExprMData__qq___closed__8_value;
static const lean_ctor_object lp_Qq_instToExprMData__qq___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_Qq_instToExprMData__qq___closed__8_value),((lean_object*)&lp_Qq_instToExprMData__qq___closed__6_value)}};
static const lean_object* lp_Qq_instToExprMData__qq___closed__9 = (const lean_object*)&lp_Qq_instToExprMData__qq___closed__9_value;
static const lean_closure_object lp_Qq_instToExprMData__qq___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_Qq_instToExprMData__qq___lam__1___boxed, .m_arity = 2, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_Qq_instToExprMData__qq___closed__9_value)} };
static const lean_object* lp_Qq_instToExprMData__qq___closed__10 = (const lean_object*)&lp_Qq_instToExprMData__qq___closed__10_value;
static const lean_ctor_object lp_Qq_instToExprMData__qq___closed__11_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_Qq_instToExprMVarId__qq___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_Qq_instToExprMData__qq___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_instToExprMData__qq___closed__11_value_aux_0),((lean_object*)&lp_Qq_instToExprMData__qq___lam__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(146, 59, 178, 55, 17, 203, 226, 154)}};
static const lean_object* lp_Qq_instToExprMData__qq___closed__11 = (const lean_object*)&lp_Qq_instToExprMData__qq___closed__11_value;
static lean_once_cell_t lp_Qq_instToExprMData__qq___closed__12_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Qq_instToExprMData__qq___closed__12;
static lean_once_cell_t lp_Qq_instToExprMData__qq___closed__13_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Qq_instToExprMData__qq___closed__13;
LEAN_EXPORT lean_object* lp_Qq_instToExprMData__qq;
static const lean_ctor_object lp_Qq_List_forIn_x27_loop___at___00toExprExpr_spec__1___redArg___closed__0_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_Qq_instToExprMVarId__qq___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_Qq_List_forIn_x27_loop___at___00toExprExpr_spec__1___redArg___closed__0_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_List_forIn_x27_loop___at___00toExprExpr_spec__1___redArg___closed__0_value_aux_0),((lean_object*)&lp_Qq_instToExprMData__qq___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(151, 95, 92, 183, 154, 214, 93, 213)}};
static const lean_ctor_object lp_Qq_List_forIn_x27_loop___at___00toExprExpr_spec__1___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_List_forIn_x27_loop___at___00toExprExpr_spec__1___redArg___closed__0_value_aux_1),((lean_object*)&lp_Qq_instToExprMData__qq___lam__0___closed__1_value),LEAN_SCALAR_PTR_LITERAL(189, 35, 57, 164, 139, 58, 100, 137)}};
static const lean_object* lp_Qq_List_forIn_x27_loop___at___00toExprExpr_spec__1___redArg___closed__0 = (const lean_object*)&lp_Qq_List_forIn_x27_loop___at___00toExprExpr_spec__1___redArg___closed__0_value;
static lean_once_cell_t lp_Qq_List_forIn_x27_loop___at___00toExprExpr_spec__1___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Qq_List_forIn_x27_loop___at___00toExprExpr_spec__1___redArg___closed__1;
static const lean_ctor_object lp_Qq_List_forIn_x27_loop___at___00toExprExpr_spec__1___redArg___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_Qq_instToExprMVarId__qq___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_Qq_List_forIn_x27_loop___at___00toExprExpr_spec__1___redArg___closed__2_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_List_forIn_x27_loop___at___00toExprExpr_spec__1___redArg___closed__2_value_aux_0),((lean_object*)&lp_Qq_instToExprMData__qq___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(151, 95, 92, 183, 154, 214, 93, 213)}};
static const lean_ctor_object lp_Qq_List_forIn_x27_loop___at___00toExprExpr_spec__1___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_List_forIn_x27_loop___at___00toExprExpr_spec__1___redArg___closed__2_value_aux_1),((lean_object*)&lp_Qq_instToExprMData__qq___lam__0___closed__2_value),LEAN_SCALAR_PTR_LITERAL(16, 228, 239, 150, 219, 122, 206, 148)}};
static const lean_object* lp_Qq_List_forIn_x27_loop___at___00toExprExpr_spec__1___redArg___closed__2 = (const lean_object*)&lp_Qq_List_forIn_x27_loop___at___00toExprExpr_spec__1___redArg___closed__2_value;
static lean_once_cell_t lp_Qq_List_forIn_x27_loop___at___00toExprExpr_spec__1___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Qq_List_forIn_x27_loop___at___00toExprExpr_spec__1___redArg___closed__3;
static lean_once_cell_t lp_Qq_List_forIn_x27_loop___at___00toExprExpr_spec__1___redArg___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Qq_List_forIn_x27_loop___at___00toExprExpr_spec__1___redArg___closed__4;
static lean_once_cell_t lp_Qq_List_forIn_x27_loop___at___00toExprExpr_spec__1___redArg___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Qq_List_forIn_x27_loop___at___00toExprExpr_spec__1___redArg___closed__5;
static const lean_ctor_object lp_Qq_List_forIn_x27_loop___at___00toExprExpr_spec__1___redArg___closed__6_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_Qq_instToExprMVarId__qq___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_Qq_List_forIn_x27_loop___at___00toExprExpr_spec__1___redArg___closed__6_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_List_forIn_x27_loop___at___00toExprExpr_spec__1___redArg___closed__6_value_aux_0),((lean_object*)&lp_Qq_instToExprMData__qq___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(151, 95, 92, 183, 154, 214, 93, 213)}};
static const lean_ctor_object lp_Qq_List_forIn_x27_loop___at___00toExprExpr_spec__1___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_List_forIn_x27_loop___at___00toExprExpr_spec__1___redArg___closed__6_value_aux_1),((lean_object*)&lp_Qq_instToExprMData__qq___lam__0___closed__8_value),LEAN_SCALAR_PTR_LITERAL(89, 147, 223, 143, 38, 139, 139, 74)}};
static const lean_object* lp_Qq_List_forIn_x27_loop___at___00toExprExpr_spec__1___redArg___closed__6 = (const lean_object*)&lp_Qq_List_forIn_x27_loop___at___00toExprExpr_spec__1___redArg___closed__6_value;
static lean_once_cell_t lp_Qq_List_forIn_x27_loop___at___00toExprExpr_spec__1___redArg___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Qq_List_forIn_x27_loop___at___00toExprExpr_spec__1___redArg___closed__7;
static const lean_ctor_object lp_Qq_List_forIn_x27_loop___at___00toExprExpr_spec__1___redArg___closed__8_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_Qq_instToExprMVarId__qq___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_Qq_List_forIn_x27_loop___at___00toExprExpr_spec__1___redArg___closed__8_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_List_forIn_x27_loop___at___00toExprExpr_spec__1___redArg___closed__8_value_aux_0),((lean_object*)&lp_Qq_instToExprMData__qq___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(151, 95, 92, 183, 154, 214, 93, 213)}};
static const lean_ctor_object lp_Qq_List_forIn_x27_loop___at___00toExprExpr_spec__1___redArg___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_List_forIn_x27_loop___at___00toExprExpr_spec__1___redArg___closed__8_value_aux_1),((lean_object*)&lp_Qq_instToExprMData__qq___lam__0___closed__9_value),LEAN_SCALAR_PTR_LITERAL(158, 235, 43, 39, 180, 27, 224, 41)}};
static const lean_object* lp_Qq_List_forIn_x27_loop___at___00toExprExpr_spec__1___redArg___closed__8 = (const lean_object*)&lp_Qq_List_forIn_x27_loop___at___00toExprExpr_spec__1___redArg___closed__8_value;
static lean_once_cell_t lp_Qq_List_forIn_x27_loop___at___00toExprExpr_spec__1___redArg___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Qq_List_forIn_x27_loop___at___00toExprExpr_spec__1___redArg___closed__9;
static const lean_ctor_object lp_Qq_List_forIn_x27_loop___at___00toExprExpr_spec__1___redArg___closed__10_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_Qq_instToExprMVarId__qq___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_Qq_List_forIn_x27_loop___at___00toExprExpr_spec__1___redArg___closed__10_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_List_forIn_x27_loop___at___00toExprExpr_spec__1___redArg___closed__10_value_aux_0),((lean_object*)&lp_Qq_instToExprMData__qq___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(151, 95, 92, 183, 154, 214, 93, 213)}};
static const lean_ctor_object lp_Qq_List_forIn_x27_loop___at___00toExprExpr_spec__1___redArg___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_List_forIn_x27_loop___at___00toExprExpr_spec__1___redArg___closed__10_value_aux_1),((lean_object*)&lp_Qq_instToExprMData__qq___lam__0___closed__10_value),LEAN_SCALAR_PTR_LITERAL(59, 229, 98, 11, 115, 199, 151, 183)}};
static const lean_object* lp_Qq_List_forIn_x27_loop___at___00toExprExpr_spec__1___redArg___closed__10 = (const lean_object*)&lp_Qq_List_forIn_x27_loop___at___00toExprExpr_spec__1___redArg___closed__10_value;
static lean_once_cell_t lp_Qq_List_forIn_x27_loop___at___00toExprExpr_spec__1___redArg___closed__11_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Qq_List_forIn_x27_loop___at___00toExprExpr_spec__1___redArg___closed__11;
static lean_once_cell_t lp_Qq_List_forIn_x27_loop___at___00toExprExpr_spec__1___redArg___closed__12_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Qq_List_forIn_x27_loop___at___00toExprExpr_spec__1___redArg___closed__12;
static lean_once_cell_t lp_Qq_List_forIn_x27_loop___at___00toExprExpr_spec__1___redArg___closed__13_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Qq_List_forIn_x27_loop___at___00toExprExpr_spec__1___redArg___closed__13;
static lean_once_cell_t lp_Qq_List_forIn_x27_loop___at___00toExprExpr_spec__1___redArg___closed__14_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Qq_List_forIn_x27_loop___at___00toExprExpr_spec__1___redArg___closed__14;
static lean_once_cell_t lp_Qq_List_forIn_x27_loop___at___00toExprExpr_spec__1___redArg___closed__15_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Qq_List_forIn_x27_loop___at___00toExprExpr_spec__1___redArg___closed__15;
LEAN_EXPORT lean_object* lp_Qq_List_forIn_x27_loop___at___00toExprExpr_spec__1___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_List_forIn_x27_loop___at___00toExprExpr_spec__1___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00toExprExpr_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00toExprExpr_spec__0___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_Qq_toExprExpr___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Expr"};
static const lean_object* lp_Qq_toExprExpr___closed__0 = (const lean_object*)&lp_Qq_toExprExpr___closed__0_value;
static const lean_string_object lp_Qq_toExprExpr___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "bvar"};
static const lean_object* lp_Qq_toExprExpr___closed__1 = (const lean_object*)&lp_Qq_toExprExpr___closed__1_value;
static const lean_ctor_object lp_Qq_toExprExpr___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_Qq_instToExprMVarId__qq___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_Qq_toExprExpr___closed__2_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_toExprExpr___closed__2_value_aux_0),((lean_object*)&lp_Qq_toExprExpr___closed__0_value),LEAN_SCALAR_PTR_LITERAL(84, 208, 74, 211, 93, 83, 88, 82)}};
static const lean_ctor_object lp_Qq_toExprExpr___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_toExprExpr___closed__2_value_aux_1),((lean_object*)&lp_Qq_toExprExpr___closed__1_value),LEAN_SCALAR_PTR_LITERAL(194, 116, 189, 113, 236, 234, 204, 95)}};
static const lean_object* lp_Qq_toExprExpr___closed__2 = (const lean_object*)&lp_Qq_toExprExpr___closed__2_value;
static lean_once_cell_t lp_Qq_toExprExpr___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Qq_toExprExpr___closed__3;
static const lean_string_object lp_Qq_toExprExpr___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "fvar"};
static const lean_object* lp_Qq_toExprExpr___closed__4 = (const lean_object*)&lp_Qq_toExprExpr___closed__4_value;
static const lean_ctor_object lp_Qq_toExprExpr___closed__5_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_Qq_instToExprMVarId__qq___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_Qq_toExprExpr___closed__5_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_toExprExpr___closed__5_value_aux_0),((lean_object*)&lp_Qq_toExprExpr___closed__0_value),LEAN_SCALAR_PTR_LITERAL(84, 208, 74, 211, 93, 83, 88, 82)}};
static const lean_ctor_object lp_Qq_toExprExpr___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_toExprExpr___closed__5_value_aux_1),((lean_object*)&lp_Qq_toExprExpr___closed__4_value),LEAN_SCALAR_PTR_LITERAL(195, 84, 31, 148, 26, 167, 194, 104)}};
static const lean_object* lp_Qq_toExprExpr___closed__5 = (const lean_object*)&lp_Qq_toExprExpr___closed__5_value;
static lean_once_cell_t lp_Qq_toExprExpr___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Qq_toExprExpr___closed__6;
static const lean_string_object lp_Qq_toExprExpr___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "FVarId"};
static const lean_object* lp_Qq_toExprExpr___closed__7 = (const lean_object*)&lp_Qq_toExprExpr___closed__7_value;
static const lean_ctor_object lp_Qq_toExprExpr___closed__8_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_Qq_instToExprMVarId__qq___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_Qq_toExprExpr___closed__8_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_toExprExpr___closed__8_value_aux_0),((lean_object*)&lp_Qq_toExprExpr___closed__7_value),LEAN_SCALAR_PTR_LITERAL(134, 80, 170, 214, 218, 146, 55, 86)}};
static const lean_ctor_object lp_Qq_toExprExpr___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_toExprExpr___closed__8_value_aux_1),((lean_object*)&lp_Qq_instToExprMVarId__qq___lam__0___closed__2_value),LEAN_SCALAR_PTR_LITERAL(246, 212, 153, 136, 172, 214, 179, 96)}};
static const lean_object* lp_Qq_toExprExpr___closed__8 = (const lean_object*)&lp_Qq_toExprExpr___closed__8_value;
static lean_once_cell_t lp_Qq_toExprExpr___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Qq_toExprExpr___closed__9;
static const lean_ctor_object lp_Qq_toExprExpr___closed__10_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_Qq_instToExprMVarId__qq___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_Qq_toExprExpr___closed__10_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_toExprExpr___closed__10_value_aux_0),((lean_object*)&lp_Qq_toExprExpr___closed__0_value),LEAN_SCALAR_PTR_LITERAL(84, 208, 74, 211, 93, 83, 88, 82)}};
static const lean_ctor_object lp_Qq_toExprExpr___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_toExprExpr___closed__10_value_aux_1),((lean_object*)&lp_Qq_toExprLevel___closed__16_value),LEAN_SCALAR_PTR_LITERAL(28, 197, 45, 187, 18, 219, 14, 58)}};
static const lean_object* lp_Qq_toExprExpr___closed__10 = (const lean_object*)&lp_Qq_toExprExpr___closed__10_value;
static lean_once_cell_t lp_Qq_toExprExpr___closed__11_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Qq_toExprExpr___closed__11;
static const lean_string_object lp_Qq_toExprExpr___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "sort"};
static const lean_object* lp_Qq_toExprExpr___closed__12 = (const lean_object*)&lp_Qq_toExprExpr___closed__12_value;
static const lean_ctor_object lp_Qq_toExprExpr___closed__13_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_Qq_instToExprMVarId__qq___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_Qq_toExprExpr___closed__13_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_toExprExpr___closed__13_value_aux_0),((lean_object*)&lp_Qq_toExprExpr___closed__0_value),LEAN_SCALAR_PTR_LITERAL(84, 208, 74, 211, 93, 83, 88, 82)}};
static const lean_ctor_object lp_Qq_toExprExpr___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_toExprExpr___closed__13_value_aux_1),((lean_object*)&lp_Qq_toExprExpr___closed__12_value),LEAN_SCALAR_PTR_LITERAL(64, 95, 209, 188, 135, 1, 196, 95)}};
static const lean_object* lp_Qq_toExprExpr___closed__13 = (const lean_object*)&lp_Qq_toExprExpr___closed__13_value;
static lean_once_cell_t lp_Qq_toExprExpr___closed__14_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Qq_toExprExpr___closed__14;
static const lean_string_object lp_Qq_toExprExpr___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "const"};
static const lean_object* lp_Qq_toExprExpr___closed__15 = (const lean_object*)&lp_Qq_toExprExpr___closed__15_value;
static const lean_ctor_object lp_Qq_toExprExpr___closed__16_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_Qq_instToExprMVarId__qq___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_Qq_toExprExpr___closed__16_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_toExprExpr___closed__16_value_aux_0),((lean_object*)&lp_Qq_toExprExpr___closed__0_value),LEAN_SCALAR_PTR_LITERAL(84, 208, 74, 211, 93, 83, 88, 82)}};
static const lean_ctor_object lp_Qq_toExprExpr___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_toExprExpr___closed__16_value_aux_1),((lean_object*)&lp_Qq_toExprExpr___closed__15_value),LEAN_SCALAR_PTR_LITERAL(22, 248, 240, 94, 191, 251, 149, 49)}};
static const lean_object* lp_Qq_toExprExpr___closed__16 = (const lean_object*)&lp_Qq_toExprExpr___closed__16_value;
static lean_once_cell_t lp_Qq_toExprExpr___closed__17_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Qq_toExprExpr___closed__17;
static const lean_string_object lp_Qq_toExprExpr___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "List"};
static const lean_object* lp_Qq_toExprExpr___closed__18 = (const lean_object*)&lp_Qq_toExprExpr___closed__18_value;
static const lean_string_object lp_Qq_toExprExpr___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "nil"};
static const lean_object* lp_Qq_toExprExpr___closed__19 = (const lean_object*)&lp_Qq_toExprExpr___closed__19_value;
static const lean_ctor_object lp_Qq_toExprExpr___closed__20_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_Qq_toExprExpr___closed__18_value),LEAN_SCALAR_PTR_LITERAL(245, 188, 225, 225, 165, 5, 251, 132)}};
static const lean_ctor_object lp_Qq_toExprExpr___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_toExprExpr___closed__20_value_aux_0),((lean_object*)&lp_Qq_toExprExpr___closed__19_value),LEAN_SCALAR_PTR_LITERAL(90, 150, 134, 113, 145, 38, 173, 251)}};
static const lean_object* lp_Qq_toExprExpr___closed__20 = (const lean_object*)&lp_Qq_toExprExpr___closed__20_value;
static const lean_ctor_object lp_Qq_toExprExpr___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_Qq_toExprExpr___closed__21 = (const lean_object*)&lp_Qq_toExprExpr___closed__21_value;
static lean_once_cell_t lp_Qq_toExprExpr___closed__22_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Qq_toExprExpr___closed__22;
static lean_once_cell_t lp_Qq_toExprExpr___closed__23_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Qq_toExprExpr___closed__23;
static const lean_string_object lp_Qq_toExprExpr___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "cons"};
static const lean_object* lp_Qq_toExprExpr___closed__24 = (const lean_object*)&lp_Qq_toExprExpr___closed__24_value;
static const lean_ctor_object lp_Qq_toExprExpr___closed__25_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_Qq_toExprExpr___closed__18_value),LEAN_SCALAR_PTR_LITERAL(245, 188, 225, 225, 165, 5, 251, 132)}};
static const lean_ctor_object lp_Qq_toExprExpr___closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_toExprExpr___closed__25_value_aux_0),((lean_object*)&lp_Qq_toExprExpr___closed__24_value),LEAN_SCALAR_PTR_LITERAL(98, 170, 59, 223, 79, 132, 139, 119)}};
static const lean_object* lp_Qq_toExprExpr___closed__25 = (const lean_object*)&lp_Qq_toExprExpr___closed__25_value;
static lean_once_cell_t lp_Qq_toExprExpr___closed__26_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Qq_toExprExpr___closed__26;
static lean_once_cell_t lp_Qq_toExprExpr___closed__27_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Qq_toExprExpr___closed__27;
static const lean_string_object lp_Qq_toExprExpr___closed__28_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "app"};
static const lean_object* lp_Qq_toExprExpr___closed__28 = (const lean_object*)&lp_Qq_toExprExpr___closed__28_value;
static const lean_ctor_object lp_Qq_toExprExpr___closed__29_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_Qq_instToExprMVarId__qq___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_Qq_toExprExpr___closed__29_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_toExprExpr___closed__29_value_aux_0),((lean_object*)&lp_Qq_toExprExpr___closed__0_value),LEAN_SCALAR_PTR_LITERAL(84, 208, 74, 211, 93, 83, 88, 82)}};
static const lean_ctor_object lp_Qq_toExprExpr___closed__29_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_toExprExpr___closed__29_value_aux_1),((lean_object*)&lp_Qq_toExprExpr___closed__28_value),LEAN_SCALAR_PTR_LITERAL(134, 107, 4, 185, 254, 245, 50, 185)}};
static const lean_object* lp_Qq_toExprExpr___closed__29 = (const lean_object*)&lp_Qq_toExprExpr___closed__29_value;
static lean_once_cell_t lp_Qq_toExprExpr___closed__30_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Qq_toExprExpr___closed__30;
static const lean_string_object lp_Qq_toExprExpr___closed__31_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "lam"};
static const lean_object* lp_Qq_toExprExpr___closed__31 = (const lean_object*)&lp_Qq_toExprExpr___closed__31_value;
static const lean_ctor_object lp_Qq_toExprExpr___closed__32_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_Qq_instToExprMVarId__qq___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_Qq_toExprExpr___closed__32_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_toExprExpr___closed__32_value_aux_0),((lean_object*)&lp_Qq_toExprExpr___closed__0_value),LEAN_SCALAR_PTR_LITERAL(84, 208, 74, 211, 93, 83, 88, 82)}};
static const lean_ctor_object lp_Qq_toExprExpr___closed__32_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_toExprExpr___closed__32_value_aux_1),((lean_object*)&lp_Qq_toExprExpr___closed__31_value),LEAN_SCALAR_PTR_LITERAL(156, 194, 121, 61, 219, 0, 202, 155)}};
static const lean_object* lp_Qq_toExprExpr___closed__32 = (const lean_object*)&lp_Qq_toExprExpr___closed__32_value;
static lean_once_cell_t lp_Qq_toExprExpr___closed__33_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Qq_toExprExpr___closed__33;
static const lean_string_object lp_Qq_toExprExpr___closed__34_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "forallE"};
static const lean_object* lp_Qq_toExprExpr___closed__34 = (const lean_object*)&lp_Qq_toExprExpr___closed__34_value;
static const lean_ctor_object lp_Qq_toExprExpr___closed__35_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_Qq_instToExprMVarId__qq___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_Qq_toExprExpr___closed__35_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_toExprExpr___closed__35_value_aux_0),((lean_object*)&lp_Qq_toExprExpr___closed__0_value),LEAN_SCALAR_PTR_LITERAL(84, 208, 74, 211, 93, 83, 88, 82)}};
static const lean_ctor_object lp_Qq_toExprExpr___closed__35_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_toExprExpr___closed__35_value_aux_1),((lean_object*)&lp_Qq_toExprExpr___closed__34_value),LEAN_SCALAR_PTR_LITERAL(209, 174, 244, 115, 50, 19, 87, 122)}};
static const lean_object* lp_Qq_toExprExpr___closed__35 = (const lean_object*)&lp_Qq_toExprExpr___closed__35_value;
static lean_once_cell_t lp_Qq_toExprExpr___closed__36_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Qq_toExprExpr___closed__36;
static const lean_string_object lp_Qq_toExprExpr___closed__37_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "letE"};
static const lean_object* lp_Qq_toExprExpr___closed__37 = (const lean_object*)&lp_Qq_toExprExpr___closed__37_value;
static const lean_ctor_object lp_Qq_toExprExpr___closed__38_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_Qq_instToExprMVarId__qq___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_Qq_toExprExpr___closed__38_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_toExprExpr___closed__38_value_aux_0),((lean_object*)&lp_Qq_toExprExpr___closed__0_value),LEAN_SCALAR_PTR_LITERAL(84, 208, 74, 211, 93, 83, 88, 82)}};
static const lean_ctor_object lp_Qq_toExprExpr___closed__38_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_toExprExpr___closed__38_value_aux_1),((lean_object*)&lp_Qq_toExprExpr___closed__37_value),LEAN_SCALAR_PTR_LITERAL(218, 165, 179, 210, 92, 162, 150, 56)}};
static const lean_object* lp_Qq_toExprExpr___closed__38 = (const lean_object*)&lp_Qq_toExprExpr___closed__38_value;
static lean_once_cell_t lp_Qq_toExprExpr___closed__39_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Qq_toExprExpr___closed__39;
static const lean_string_object lp_Qq_toExprExpr___closed__40_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "lit"};
static const lean_object* lp_Qq_toExprExpr___closed__40 = (const lean_object*)&lp_Qq_toExprExpr___closed__40_value;
static const lean_ctor_object lp_Qq_toExprExpr___closed__41_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_Qq_instToExprMVarId__qq___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_Qq_toExprExpr___closed__41_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_toExprExpr___closed__41_value_aux_0),((lean_object*)&lp_Qq_toExprExpr___closed__0_value),LEAN_SCALAR_PTR_LITERAL(84, 208, 74, 211, 93, 83, 88, 82)}};
static const lean_ctor_object lp_Qq_toExprExpr___closed__41_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_toExprExpr___closed__41_value_aux_1),((lean_object*)&lp_Qq_toExprExpr___closed__40_value),LEAN_SCALAR_PTR_LITERAL(142, 45, 148, 16, 248, 234, 208, 241)}};
static const lean_object* lp_Qq_toExprExpr___closed__41 = (const lean_object*)&lp_Qq_toExprExpr___closed__41_value;
static lean_once_cell_t lp_Qq_toExprExpr___closed__42_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Qq_toExprExpr___closed__42;
static const lean_string_object lp_Qq_toExprExpr___closed__43_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Literal"};
static const lean_object* lp_Qq_toExprExpr___closed__43 = (const lean_object*)&lp_Qq_toExprExpr___closed__43_value;
static const lean_string_object lp_Qq_toExprExpr___closed__44_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "natVal"};
static const lean_object* lp_Qq_toExprExpr___closed__44 = (const lean_object*)&lp_Qq_toExprExpr___closed__44_value;
static const lean_ctor_object lp_Qq_toExprExpr___closed__45_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_Qq_instToExprMVarId__qq___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_Qq_toExprExpr___closed__45_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_toExprExpr___closed__45_value_aux_0),((lean_object*)&lp_Qq_toExprExpr___closed__43_value),LEAN_SCALAR_PTR_LITERAL(39, 22, 220, 12, 129, 114, 43, 97)}};
static const lean_ctor_object lp_Qq_toExprExpr___closed__45_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_toExprExpr___closed__45_value_aux_1),((lean_object*)&lp_Qq_toExprExpr___closed__44_value),LEAN_SCALAR_PTR_LITERAL(64, 199, 201, 37, 137, 51, 1, 129)}};
static const lean_object* lp_Qq_toExprExpr___closed__45 = (const lean_object*)&lp_Qq_toExprExpr___closed__45_value;
static lean_once_cell_t lp_Qq_toExprExpr___closed__46_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Qq_toExprExpr___closed__46;
static const lean_string_object lp_Qq_toExprExpr___closed__47_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "strVal"};
static const lean_object* lp_Qq_toExprExpr___closed__47 = (const lean_object*)&lp_Qq_toExprExpr___closed__47_value;
static const lean_ctor_object lp_Qq_toExprExpr___closed__48_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_Qq_instToExprMVarId__qq___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_Qq_toExprExpr___closed__48_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_toExprExpr___closed__48_value_aux_0),((lean_object*)&lp_Qq_toExprExpr___closed__43_value),LEAN_SCALAR_PTR_LITERAL(39, 22, 220, 12, 129, 114, 43, 97)}};
static const lean_ctor_object lp_Qq_toExprExpr___closed__48_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_toExprExpr___closed__48_value_aux_1),((lean_object*)&lp_Qq_toExprExpr___closed__47_value),LEAN_SCALAR_PTR_LITERAL(68, 214, 249, 146, 84, 160, 212, 27)}};
static const lean_object* lp_Qq_toExprExpr___closed__48 = (const lean_object*)&lp_Qq_toExprExpr___closed__48_value;
static lean_once_cell_t lp_Qq_toExprExpr___closed__49_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Qq_toExprExpr___closed__49;
static const lean_string_object lp_Qq_toExprExpr___closed__50_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "mdata"};
static const lean_object* lp_Qq_toExprExpr___closed__50 = (const lean_object*)&lp_Qq_toExprExpr___closed__50_value;
static const lean_ctor_object lp_Qq_toExprExpr___closed__51_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_Qq_instToExprMVarId__qq___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_Qq_toExprExpr___closed__51_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_toExprExpr___closed__51_value_aux_0),((lean_object*)&lp_Qq_toExprExpr___closed__0_value),LEAN_SCALAR_PTR_LITERAL(84, 208, 74, 211, 93, 83, 88, 82)}};
static const lean_ctor_object lp_Qq_toExprExpr___closed__51_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_toExprExpr___closed__51_value_aux_1),((lean_object*)&lp_Qq_toExprExpr___closed__50_value),LEAN_SCALAR_PTR_LITERAL(32, 170, 73, 140, 82, 239, 68, 98)}};
static const lean_object* lp_Qq_toExprExpr___closed__51 = (const lean_object*)&lp_Qq_toExprExpr___closed__51_value;
static lean_once_cell_t lp_Qq_toExprExpr___closed__52_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Qq_toExprExpr___closed__52;
static const lean_string_object lp_Qq_toExprExpr___closed__53_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "proj"};
static const lean_object* lp_Qq_toExprExpr___closed__53 = (const lean_object*)&lp_Qq_toExprExpr___closed__53_value;
static const lean_ctor_object lp_Qq_toExprExpr___closed__54_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_Qq_instToExprMVarId__qq___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_Qq_toExprExpr___closed__54_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_toExprExpr___closed__54_value_aux_0),((lean_object*)&lp_Qq_toExprExpr___closed__0_value),LEAN_SCALAR_PTR_LITERAL(84, 208, 74, 211, 93, 83, 88, 82)}};
static const lean_ctor_object lp_Qq_toExprExpr___closed__54_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_toExprExpr___closed__54_value_aux_1),((lean_object*)&lp_Qq_toExprExpr___closed__53_value),LEAN_SCALAR_PTR_LITERAL(164, 93, 179, 84, 156, 219, 121, 238)}};
static const lean_object* lp_Qq_toExprExpr___closed__54 = (const lean_object*)&lp_Qq_toExprExpr___closed__54_value;
static lean_once_cell_t lp_Qq_toExprExpr___closed__55_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Qq_toExprExpr___closed__55;
LEAN_EXPORT lean_object* lp_Qq_toExprExpr(lean_object*);
LEAN_EXPORT lean_object* lp_Qq_List_forIn_x27_loop___at___00toExprExpr_spec__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_List_forIn_x27_loop___at___00toExprExpr_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_Qq_instToExprExpr__qq___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_Qq_toExprExpr, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_Qq_instToExprExpr__qq___closed__0 = (const lean_object*)&lp_Qq_instToExprExpr__qq___closed__0_value;
static const lean_ctor_object lp_Qq_instToExprExpr__qq___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_Qq_instToExprMVarId__qq___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_Qq_instToExprExpr__qq___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_instToExprExpr__qq___closed__1_value_aux_0),((lean_object*)&lp_Qq_toExprExpr___closed__0_value),LEAN_SCALAR_PTR_LITERAL(84, 208, 74, 211, 93, 83, 88, 82)}};
static const lean_object* lp_Qq_instToExprExpr__qq___closed__1 = (const lean_object*)&lp_Qq_instToExprExpr__qq___closed__1_value;
static lean_once_cell_t lp_Qq_instToExprExpr__qq___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Qq_instToExprExpr__qq___closed__2;
static lean_once_cell_t lp_Qq_instToExprExpr__qq___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Qq_instToExprExpr__qq___closed__3;
LEAN_EXPORT lean_object* lp_Qq_instToExprExpr__qq;
static lean_object* _init_lp_Qq_instToExprMVarId__qq___lam__0___closed__4(void){
_start:
{
lean_object* v___x_8_; lean_object* v___x_9_; lean_object* v___x_10_; 
v___x_8_ = lean_box(0);
v___x_9_ = ((lean_object*)(lp_Qq_instToExprMVarId__qq___lam__0___closed__3));
v___x_10_ = l_Lean_Expr_const___override(v___x_9_, v___x_8_);
return v___x_10_;
}
}
LEAN_EXPORT lean_object* lp_Qq_instToExprMVarId__qq___lam__0(lean_object* v_i_11_){
_start:
{
lean_object* v___x_12_; lean_object* v___x_13_; lean_object* v___x_14_; 
v___x_12_ = lean_obj_once(&lp_Qq_instToExprMVarId__qq___lam__0___closed__4, &lp_Qq_instToExprMVarId__qq___lam__0___closed__4_once, _init_lp_Qq_instToExprMVarId__qq___lam__0___closed__4);
v___x_13_ = l___private_Lean_ToExpr_0__Lean_Name_toExprAux(v_i_11_);
v___x_14_ = l_Lean_Expr_app___override(v___x_12_, v___x_13_);
return v___x_14_;
}
}
static lean_object* _init_lp_Qq_instToExprMVarId__qq___closed__2(void){
_start:
{
lean_object* v___x_19_; lean_object* v___x_20_; lean_object* v___x_21_; 
v___x_19_ = lean_box(0);
v___x_20_ = ((lean_object*)(lp_Qq_instToExprMVarId__qq___closed__1));
v___x_21_ = l_Lean_Expr_const___override(v___x_20_, v___x_19_);
return v___x_21_;
}
}
static lean_object* _init_lp_Qq_instToExprMVarId__qq___closed__3(void){
_start:
{
lean_object* v___x_22_; lean_object* v___f_23_; lean_object* v___x_24_; 
v___x_22_ = lean_obj_once(&lp_Qq_instToExprMVarId__qq___closed__2, &lp_Qq_instToExprMVarId__qq___closed__2_once, _init_lp_Qq_instToExprMVarId__qq___closed__2);
v___f_23_ = ((lean_object*)(lp_Qq_instToExprMVarId__qq___closed__0));
v___x_24_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_24_, 0, v___f_23_);
lean_ctor_set(v___x_24_, 1, v___x_22_);
return v___x_24_;
}
}
static lean_object* _init_lp_Qq_instToExprMVarId__qq(void){
_start:
{
lean_object* v___x_25_; 
v___x_25_ = lean_obj_once(&lp_Qq_instToExprMVarId__qq___closed__3, &lp_Qq_instToExprMVarId__qq___closed__3_once, _init_lp_Qq_instToExprMVarId__qq___closed__3);
return v___x_25_;
}
}
static lean_object* _init_lp_Qq_instToExprLevelMVarId__qq___lam__0___closed__2(void){
_start:
{
lean_object* v___x_31_; lean_object* v___x_32_; lean_object* v___x_33_; 
v___x_31_ = lean_box(0);
v___x_32_ = ((lean_object*)(lp_Qq_instToExprLevelMVarId__qq___lam__0___closed__1));
v___x_33_ = l_Lean_Expr_const___override(v___x_32_, v___x_31_);
return v___x_33_;
}
}
LEAN_EXPORT lean_object* lp_Qq_instToExprLevelMVarId__qq___lam__0(lean_object* v_i_34_){
_start:
{
lean_object* v___x_35_; lean_object* v___x_36_; lean_object* v___x_37_; 
v___x_35_ = lean_obj_once(&lp_Qq_instToExprLevelMVarId__qq___lam__0___closed__2, &lp_Qq_instToExprLevelMVarId__qq___lam__0___closed__2_once, _init_lp_Qq_instToExprLevelMVarId__qq___lam__0___closed__2);
v___x_36_ = l___private_Lean_ToExpr_0__Lean_Name_toExprAux(v_i_34_);
v___x_37_ = l_Lean_Expr_app___override(v___x_35_, v___x_36_);
return v___x_37_;
}
}
static lean_object* _init_lp_Qq_instToExprLevelMVarId__qq___closed__2(void){
_start:
{
lean_object* v___x_42_; lean_object* v___x_43_; lean_object* v___x_44_; 
v___x_42_ = lean_box(0);
v___x_43_ = ((lean_object*)(lp_Qq_instToExprLevelMVarId__qq___closed__1));
v___x_44_ = l_Lean_Expr_const___override(v___x_43_, v___x_42_);
return v___x_44_;
}
}
static lean_object* _init_lp_Qq_instToExprLevelMVarId__qq___closed__3(void){
_start:
{
lean_object* v___x_45_; lean_object* v___f_46_; lean_object* v___x_47_; 
v___x_45_ = lean_obj_once(&lp_Qq_instToExprLevelMVarId__qq___closed__2, &lp_Qq_instToExprLevelMVarId__qq___closed__2_once, _init_lp_Qq_instToExprLevelMVarId__qq___closed__2);
v___f_46_ = ((lean_object*)(lp_Qq_instToExprLevelMVarId__qq___closed__0));
v___x_47_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_47_, 0, v___f_46_);
lean_ctor_set(v___x_47_, 1, v___x_45_);
return v___x_47_;
}
}
static lean_object* _init_lp_Qq_instToExprLevelMVarId__qq(void){
_start:
{
lean_object* v___x_48_; 
v___x_48_ = lean_obj_once(&lp_Qq_instToExprLevelMVarId__qq___closed__3, &lp_Qq_instToExprLevelMVarId__qq___closed__3_once, _init_lp_Qq_instToExprLevelMVarId__qq___closed__3);
return v___x_48_;
}
}
static lean_object* _init_lp_Qq_toExprLevel___closed__3(void){
_start:
{
lean_object* v___x_55_; lean_object* v___x_56_; lean_object* v___x_57_; 
v___x_55_ = lean_box(0);
v___x_56_ = ((lean_object*)(lp_Qq_toExprLevel___closed__2));
v___x_57_ = l_Lean_Expr_const___override(v___x_56_, v___x_55_);
return v___x_57_;
}
}
static lean_object* _init_lp_Qq_toExprLevel___closed__6(void){
_start:
{
lean_object* v___x_63_; lean_object* v___x_64_; lean_object* v___x_65_; 
v___x_63_ = lean_box(0);
v___x_64_ = ((lean_object*)(lp_Qq_toExprLevel___closed__5));
v___x_65_ = l_Lean_Expr_const___override(v___x_64_, v___x_63_);
return v___x_65_;
}
}
static lean_object* _init_lp_Qq_toExprLevel___closed__9(void){
_start:
{
lean_object* v___x_71_; lean_object* v___x_72_; lean_object* v___x_73_; 
v___x_71_ = lean_box(0);
v___x_72_ = ((lean_object*)(lp_Qq_toExprLevel___closed__8));
v___x_73_ = l_Lean_Expr_const___override(v___x_72_, v___x_71_);
return v___x_73_;
}
}
static lean_object* _init_lp_Qq_toExprLevel___closed__12(void){
_start:
{
lean_object* v___x_79_; lean_object* v___x_80_; lean_object* v___x_81_; 
v___x_79_ = lean_box(0);
v___x_80_ = ((lean_object*)(lp_Qq_toExprLevel___closed__11));
v___x_81_ = l_Lean_Expr_const___override(v___x_80_, v___x_79_);
return v___x_81_;
}
}
static lean_object* _init_lp_Qq_toExprLevel___closed__15(void){
_start:
{
lean_object* v___x_87_; lean_object* v___x_88_; lean_object* v___x_89_; 
v___x_87_ = lean_box(0);
v___x_88_ = ((lean_object*)(lp_Qq_toExprLevel___closed__14));
v___x_89_ = l_Lean_Expr_const___override(v___x_88_, v___x_87_);
return v___x_89_;
}
}
static lean_object* _init_lp_Qq_toExprLevel___closed__18(void){
_start:
{
lean_object* v___x_95_; lean_object* v___x_96_; lean_object* v___x_97_; 
v___x_95_ = lean_box(0);
v___x_96_ = ((lean_object*)(lp_Qq_toExprLevel___closed__17));
v___x_97_ = l_Lean_Expr_const___override(v___x_96_, v___x_95_);
return v___x_97_;
}
}
LEAN_EXPORT lean_object* lp_Qq_toExprLevel(lean_object* v_x_98_){
_start:
{
switch(lean_obj_tag(v_x_98_))
{
case 0:
{
lean_object* v___x_99_; 
v___x_99_ = lean_obj_once(&lp_Qq_toExprLevel___closed__3, &lp_Qq_toExprLevel___closed__3_once, _init_lp_Qq_toExprLevel___closed__3);
return v___x_99_;
}
case 1:
{
lean_object* v_a_100_; lean_object* v___x_101_; lean_object* v___x_102_; lean_object* v___x_103_; 
v_a_100_ = lean_ctor_get(v_x_98_, 0);
lean_inc(v_a_100_);
lean_dec_ref_known(v_x_98_, 1);
v___x_101_ = lean_obj_once(&lp_Qq_toExprLevel___closed__6, &lp_Qq_toExprLevel___closed__6_once, _init_lp_Qq_toExprLevel___closed__6);
v___x_102_ = lp_Qq_toExprLevel(v_a_100_);
v___x_103_ = l_Lean_Expr_app___override(v___x_101_, v___x_102_);
return v___x_103_;
}
case 2:
{
lean_object* v_a_104_; lean_object* v_a_105_; lean_object* v___x_106_; lean_object* v___x_107_; lean_object* v___x_108_; lean_object* v___x_109_; 
v_a_104_ = lean_ctor_get(v_x_98_, 0);
lean_inc(v_a_104_);
v_a_105_ = lean_ctor_get(v_x_98_, 1);
lean_inc(v_a_105_);
lean_dec_ref_known(v_x_98_, 2);
v___x_106_ = lean_obj_once(&lp_Qq_toExprLevel___closed__9, &lp_Qq_toExprLevel___closed__9_once, _init_lp_Qq_toExprLevel___closed__9);
v___x_107_ = lp_Qq_toExprLevel(v_a_104_);
v___x_108_ = lp_Qq_toExprLevel(v_a_105_);
v___x_109_ = l_Lean_mkAppB(v___x_106_, v___x_107_, v___x_108_);
return v___x_109_;
}
case 3:
{
lean_object* v_a_110_; lean_object* v_a_111_; lean_object* v___x_112_; lean_object* v___x_113_; lean_object* v___x_114_; lean_object* v___x_115_; 
v_a_110_ = lean_ctor_get(v_x_98_, 0);
lean_inc(v_a_110_);
v_a_111_ = lean_ctor_get(v_x_98_, 1);
lean_inc(v_a_111_);
lean_dec_ref_known(v_x_98_, 2);
v___x_112_ = lean_obj_once(&lp_Qq_toExprLevel___closed__12, &lp_Qq_toExprLevel___closed__12_once, _init_lp_Qq_toExprLevel___closed__12);
v___x_113_ = lp_Qq_toExprLevel(v_a_110_);
v___x_114_ = lp_Qq_toExprLevel(v_a_111_);
v___x_115_ = l_Lean_mkAppB(v___x_112_, v___x_113_, v___x_114_);
return v___x_115_;
}
case 4:
{
lean_object* v_a_116_; lean_object* v___x_117_; lean_object* v___x_118_; lean_object* v___x_119_; 
v_a_116_ = lean_ctor_get(v_x_98_, 0);
lean_inc(v_a_116_);
lean_dec_ref_known(v_x_98_, 1);
v___x_117_ = lean_obj_once(&lp_Qq_toExprLevel___closed__15, &lp_Qq_toExprLevel___closed__15_once, _init_lp_Qq_toExprLevel___closed__15);
v___x_118_ = l___private_Lean_ToExpr_0__Lean_Name_toExprAux(v_a_116_);
v___x_119_ = l_Lean_Expr_app___override(v___x_117_, v___x_118_);
return v___x_119_;
}
default: 
{
lean_object* v_a_120_; lean_object* v___x_121_; lean_object* v___x_122_; lean_object* v___x_123_; lean_object* v___x_124_; lean_object* v___x_125_; 
v_a_120_ = lean_ctor_get(v_x_98_, 0);
lean_inc(v_a_120_);
lean_dec_ref_known(v_x_98_, 1);
v___x_121_ = lean_obj_once(&lp_Qq_toExprLevel___closed__18, &lp_Qq_toExprLevel___closed__18_once, _init_lp_Qq_toExprLevel___closed__18);
v___x_122_ = lean_obj_once(&lp_Qq_instToExprLevelMVarId__qq___lam__0___closed__2, &lp_Qq_instToExprLevelMVarId__qq___lam__0___closed__2_once, _init_lp_Qq_instToExprLevelMVarId__qq___lam__0___closed__2);
v___x_123_ = l___private_Lean_ToExpr_0__Lean_Name_toExprAux(v_a_120_);
v___x_124_ = l_Lean_Expr_app___override(v___x_122_, v___x_123_);
v___x_125_ = l_Lean_Expr_app___override(v___x_121_, v___x_124_);
return v___x_125_;
}
}
}
}
static lean_object* _init_lp_Qq_instToExprLevel__qq___closed__2(void){
_start:
{
lean_object* v___x_130_; lean_object* v___x_131_; lean_object* v___x_132_; 
v___x_130_ = lean_box(0);
v___x_131_ = ((lean_object*)(lp_Qq_instToExprLevel__qq___closed__1));
v___x_132_ = l_Lean_Expr_const___override(v___x_131_, v___x_130_);
return v___x_132_;
}
}
static lean_object* _init_lp_Qq_instToExprLevel__qq___closed__3(void){
_start:
{
lean_object* v___x_133_; lean_object* v___x_134_; lean_object* v___x_135_; 
v___x_133_ = lean_obj_once(&lp_Qq_instToExprLevel__qq___closed__2, &lp_Qq_instToExprLevel__qq___closed__2_once, _init_lp_Qq_instToExprLevel__qq___closed__2);
v___x_134_ = ((lean_object*)(lp_Qq_instToExprLevel__qq___closed__0));
v___x_135_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_135_, 0, v___x_134_);
lean_ctor_set(v___x_135_, 1, v___x_133_);
return v___x_135_;
}
}
static lean_object* _init_lp_Qq_instToExprLevel__qq(void){
_start:
{
lean_object* v___x_136_; 
v___x_136_ = lean_obj_once(&lp_Qq_instToExprLevel__qq___closed__3, &lp_Qq_instToExprLevel__qq___closed__3_once, _init_lp_Qq_instToExprLevel__qq___closed__3);
return v___x_136_;
}
}
static lean_object* _init_lp_Qq_instToExprBinderInfo__qq___lam__0___closed__3(void){
_start:
{
lean_object* v___x_143_; lean_object* v___x_144_; lean_object* v___x_145_; 
v___x_143_ = lean_box(0);
v___x_144_ = ((lean_object*)(lp_Qq_instToExprBinderInfo__qq___lam__0___closed__2));
v___x_145_ = l_Lean_Expr_const___override(v___x_144_, v___x_143_);
return v___x_145_;
}
}
static lean_object* _init_lp_Qq_instToExprBinderInfo__qq___lam__0___closed__6(void){
_start:
{
lean_object* v___x_151_; lean_object* v___x_152_; lean_object* v___x_153_; 
v___x_151_ = lean_box(0);
v___x_152_ = ((lean_object*)(lp_Qq_instToExprBinderInfo__qq___lam__0___closed__5));
v___x_153_ = l_Lean_Expr_const___override(v___x_152_, v___x_151_);
return v___x_153_;
}
}
static lean_object* _init_lp_Qq_instToExprBinderInfo__qq___lam__0___closed__9(void){
_start:
{
lean_object* v___x_159_; lean_object* v___x_160_; lean_object* v___x_161_; 
v___x_159_ = lean_box(0);
v___x_160_ = ((lean_object*)(lp_Qq_instToExprBinderInfo__qq___lam__0___closed__8));
v___x_161_ = l_Lean_Expr_const___override(v___x_160_, v___x_159_);
return v___x_161_;
}
}
static lean_object* _init_lp_Qq_instToExprBinderInfo__qq___lam__0___closed__12(void){
_start:
{
lean_object* v___x_167_; lean_object* v___x_168_; lean_object* v___x_169_; 
v___x_167_ = lean_box(0);
v___x_168_ = ((lean_object*)(lp_Qq_instToExprBinderInfo__qq___lam__0___closed__11));
v___x_169_ = l_Lean_Expr_const___override(v___x_168_, v___x_167_);
return v___x_169_;
}
}
LEAN_EXPORT lean_object* lp_Qq_instToExprBinderInfo__qq___lam__0(uint8_t v_bi_170_){
_start:
{
switch(v_bi_170_)
{
case 0:
{
lean_object* v___x_171_; 
v___x_171_ = lean_obj_once(&lp_Qq_instToExprBinderInfo__qq___lam__0___closed__3, &lp_Qq_instToExprBinderInfo__qq___lam__0___closed__3_once, _init_lp_Qq_instToExprBinderInfo__qq___lam__0___closed__3);
return v___x_171_;
}
case 1:
{
lean_object* v___x_172_; 
v___x_172_ = lean_obj_once(&lp_Qq_instToExprBinderInfo__qq___lam__0___closed__6, &lp_Qq_instToExprBinderInfo__qq___lam__0___closed__6_once, _init_lp_Qq_instToExprBinderInfo__qq___lam__0___closed__6);
return v___x_172_;
}
case 2:
{
lean_object* v___x_173_; 
v___x_173_ = lean_obj_once(&lp_Qq_instToExprBinderInfo__qq___lam__0___closed__9, &lp_Qq_instToExprBinderInfo__qq___lam__0___closed__9_once, _init_lp_Qq_instToExprBinderInfo__qq___lam__0___closed__9);
return v___x_173_;
}
default: 
{
lean_object* v___x_174_; 
v___x_174_ = lean_obj_once(&lp_Qq_instToExprBinderInfo__qq___lam__0___closed__12, &lp_Qq_instToExprBinderInfo__qq___lam__0___closed__12_once, _init_lp_Qq_instToExprBinderInfo__qq___lam__0___closed__12);
return v___x_174_;
}
}
}
}
LEAN_EXPORT lean_object* lp_Qq_instToExprBinderInfo__qq___lam__0___boxed(lean_object* v_bi_175_){
_start:
{
uint8_t v_bi_boxed_176_; lean_object* v_res_177_; 
v_bi_boxed_176_ = lean_unbox(v_bi_175_);
v_res_177_ = lp_Qq_instToExprBinderInfo__qq___lam__0(v_bi_boxed_176_);
return v_res_177_;
}
}
static lean_object* _init_lp_Qq_instToExprBinderInfo__qq___closed__2(void){
_start:
{
lean_object* v___x_182_; lean_object* v___x_183_; lean_object* v___x_184_; 
v___x_182_ = lean_box(0);
v___x_183_ = ((lean_object*)(lp_Qq_instToExprBinderInfo__qq___closed__1));
v___x_184_ = l_Lean_Expr_const___override(v___x_183_, v___x_182_);
return v___x_184_;
}
}
static lean_object* _init_lp_Qq_instToExprBinderInfo__qq___closed__3(void){
_start:
{
lean_object* v___x_185_; lean_object* v___f_186_; lean_object* v___x_187_; 
v___x_185_ = lean_obj_once(&lp_Qq_instToExprBinderInfo__qq___closed__2, &lp_Qq_instToExprBinderInfo__qq___closed__2_once, _init_lp_Qq_instToExprBinderInfo__qq___closed__2);
v___f_186_ = ((lean_object*)(lp_Qq_instToExprBinderInfo__qq___closed__0));
v___x_187_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_187_, 0, v___f_186_);
lean_ctor_set(v___x_187_, 1, v___x_185_);
return v___x_187_;
}
}
static lean_object* _init_lp_Qq_instToExprBinderInfo__qq(void){
_start:
{
lean_object* v___x_188_; 
v___x_188_ = lean_obj_once(&lp_Qq_instToExprBinderInfo__qq___closed__3, &lp_Qq_instToExprBinderInfo__qq___closed__3_once, _init_lp_Qq_instToExprBinderInfo__qq___closed__3);
return v___x_188_;
}
}
static lean_object* _init_lp_Qq_instToExprMData__qq___lam__0___closed__11(void){
_start:
{
lean_object* v___x_204_; lean_object* v___x_205_; 
v___x_204_ = lean_unsigned_to_nat(0u);
v___x_205_ = lean_nat_to_int(v___x_204_);
return v___x_205_;
}
}
static lean_object* _init_lp_Qq_instToExprMData__qq___lam__0___closed__15(void){
_start:
{
lean_object* v___x_211_; lean_object* v___x_212_; 
v___x_211_ = lean_unsigned_to_nat(0u);
v___x_212_ = l_Lean_Level_ofNat(v___x_211_);
return v___x_212_;
}
}
LEAN_EXPORT lean_object* lp_Qq_instToExprMData__qq___lam__0(lean_object* v___x_220_, lean_object* v___x_221_, lean_object* v_a_222_, lean_object* v_x_223_, lean_object* v___y_224_){
_start:
{
lean_object* v_fst_225_; lean_object* v_snd_226_; lean_object* v___x_228_; uint8_t v_isShared_229_; uint8_t v_isSharedCheck_329_; 
v_fst_225_ = lean_ctor_get(v_a_222_, 0);
v_snd_226_ = lean_ctor_get(v_a_222_, 1);
v_isSharedCheck_329_ = !lean_is_exclusive(v_a_222_);
if (v_isSharedCheck_329_ == 0)
{
v___x_228_ = v_a_222_;
v_isShared_229_ = v_isSharedCheck_329_;
goto v_resetjp_227_;
}
else
{
lean_inc(v_snd_226_);
lean_inc(v_fst_225_);
lean_dec(v_a_222_);
v___x_228_ = lean_box(0);
v_isShared_229_ = v_isSharedCheck_329_;
goto v_resetjp_227_;
}
v_resetjp_227_:
{
lean_object* v___x_230_; 
v___x_230_ = l___private_Lean_ToExpr_0__Lean_Name_toExprAux(v_fst_225_);
switch(lean_obj_tag(v_snd_226_))
{
case 0:
{
lean_object* v_v_231_; lean_object* v___x_233_; uint8_t v_isShared_234_; uint8_t v_isSharedCheck_244_; 
lean_del_object(v___x_228_);
v_v_231_ = lean_ctor_get(v_snd_226_, 0);
v_isSharedCheck_244_ = !lean_is_exclusive(v_snd_226_);
if (v_isSharedCheck_244_ == 0)
{
v___x_233_ = v_snd_226_;
v_isShared_234_ = v_isSharedCheck_244_;
goto v_resetjp_232_;
}
else
{
lean_inc(v_v_231_);
lean_dec(v_snd_226_);
v___x_233_ = lean_box(0);
v_isShared_234_ = v_isSharedCheck_244_;
goto v_resetjp_232_;
}
v_resetjp_232_:
{
lean_object* v___x_235_; lean_object* v___x_236_; lean_object* v___x_237_; lean_object* v___x_238_; lean_object* v___x_239_; lean_object* v___x_240_; lean_object* v___x_242_; 
v___x_235_ = ((lean_object*)(lp_Qq_instToExprMData__qq___lam__0___closed__0));
v___x_236_ = ((lean_object*)(lp_Qq_instToExprMData__qq___lam__0___closed__1));
v___x_237_ = l_Lean_Name_mkStr3(v___x_220_, v___x_235_, v___x_236_);
v___x_238_ = l_Lean_Expr_const___override(v___x_237_, v___x_221_);
v___x_239_ = l_Lean_mkStrLit(v_v_231_);
v___x_240_ = l_Lean_mkApp3(v___x_238_, v___y_224_, v___x_230_, v___x_239_);
if (v_isShared_234_ == 0)
{
lean_ctor_set_tag(v___x_233_, 1);
lean_ctor_set(v___x_233_, 0, v___x_240_);
v___x_242_ = v___x_233_;
goto v_reusejp_241_;
}
else
{
lean_object* v_reuseFailAlloc_243_; 
v_reuseFailAlloc_243_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_243_, 0, v___x_240_);
v___x_242_ = v_reuseFailAlloc_243_;
goto v_reusejp_241_;
}
v_reusejp_241_:
{
return v___x_242_;
}
}
}
case 1:
{
uint8_t v_v_245_; lean_object* v___x_246_; lean_object* v___x_247_; lean_object* v___x_248_; lean_object* v___x_249_; 
lean_del_object(v___x_228_);
v_v_245_ = lean_ctor_get_uint8(v_snd_226_, 0);
lean_dec_ref_known(v_snd_226_, 0);
v___x_246_ = ((lean_object*)(lp_Qq_instToExprMData__qq___lam__0___closed__0));
v___x_247_ = ((lean_object*)(lp_Qq_instToExprMData__qq___lam__0___closed__2));
v___x_248_ = l_Lean_Name_mkStr3(v___x_220_, v___x_246_, v___x_247_);
lean_inc(v___x_221_);
v___x_249_ = l_Lean_Expr_const___override(v___x_248_, v___x_221_);
if (v_v_245_ == 0)
{
lean_object* v___x_250_; lean_object* v___x_251_; lean_object* v___x_252_; lean_object* v___x_253_; 
v___x_250_ = ((lean_object*)(lp_Qq_instToExprMData__qq___lam__0___closed__5));
v___x_251_ = l_Lean_mkConst(v___x_250_, v___x_221_);
v___x_252_ = l_Lean_mkApp3(v___x_249_, v___y_224_, v___x_230_, v___x_251_);
v___x_253_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_253_, 0, v___x_252_);
return v___x_253_;
}
else
{
lean_object* v___x_254_; lean_object* v___x_255_; lean_object* v___x_256_; lean_object* v___x_257_; 
v___x_254_ = ((lean_object*)(lp_Qq_instToExprMData__qq___lam__0___closed__7));
v___x_255_ = l_Lean_mkConst(v___x_254_, v___x_221_);
v___x_256_ = l_Lean_mkApp3(v___x_249_, v___y_224_, v___x_230_, v___x_255_);
v___x_257_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_257_, 0, v___x_256_);
return v___x_257_;
}
}
case 2:
{
lean_object* v_v_258_; lean_object* v___x_260_; uint8_t v_isShared_261_; uint8_t v_isSharedCheck_271_; 
lean_del_object(v___x_228_);
v_v_258_ = lean_ctor_get(v_snd_226_, 0);
v_isSharedCheck_271_ = !lean_is_exclusive(v_snd_226_);
if (v_isSharedCheck_271_ == 0)
{
v___x_260_ = v_snd_226_;
v_isShared_261_ = v_isSharedCheck_271_;
goto v_resetjp_259_;
}
else
{
lean_inc(v_v_258_);
lean_dec(v_snd_226_);
v___x_260_ = lean_box(0);
v_isShared_261_ = v_isSharedCheck_271_;
goto v_resetjp_259_;
}
v_resetjp_259_:
{
lean_object* v___x_262_; lean_object* v___x_263_; lean_object* v___x_264_; lean_object* v___x_265_; lean_object* v___x_266_; lean_object* v___x_267_; lean_object* v___x_269_; 
v___x_262_ = ((lean_object*)(lp_Qq_instToExprMData__qq___lam__0___closed__0));
v___x_263_ = ((lean_object*)(lp_Qq_instToExprMData__qq___lam__0___closed__8));
v___x_264_ = l_Lean_Name_mkStr3(v___x_220_, v___x_262_, v___x_263_);
v___x_265_ = l_Lean_Expr_const___override(v___x_264_, v___x_221_);
v___x_266_ = l___private_Lean_ToExpr_0__Lean_Name_toExprAux(v_v_258_);
v___x_267_ = l_Lean_mkApp3(v___x_265_, v___y_224_, v___x_230_, v___x_266_);
if (v_isShared_261_ == 0)
{
lean_ctor_set_tag(v___x_260_, 1);
lean_ctor_set(v___x_260_, 0, v___x_267_);
v___x_269_ = v___x_260_;
goto v_reusejp_268_;
}
else
{
lean_object* v_reuseFailAlloc_270_; 
v_reuseFailAlloc_270_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_270_, 0, v___x_267_);
v___x_269_ = v_reuseFailAlloc_270_;
goto v_reusejp_268_;
}
v_reusejp_268_:
{
return v___x_269_;
}
}
}
case 3:
{
lean_object* v_v_272_; lean_object* v___x_274_; uint8_t v_isShared_275_; uint8_t v_isSharedCheck_285_; 
lean_del_object(v___x_228_);
v_v_272_ = lean_ctor_get(v_snd_226_, 0);
v_isSharedCheck_285_ = !lean_is_exclusive(v_snd_226_);
if (v_isSharedCheck_285_ == 0)
{
v___x_274_ = v_snd_226_;
v_isShared_275_ = v_isSharedCheck_285_;
goto v_resetjp_273_;
}
else
{
lean_inc(v_v_272_);
lean_dec(v_snd_226_);
v___x_274_ = lean_box(0);
v_isShared_275_ = v_isSharedCheck_285_;
goto v_resetjp_273_;
}
v_resetjp_273_:
{
lean_object* v___x_276_; lean_object* v___x_277_; lean_object* v___x_278_; lean_object* v___x_279_; lean_object* v___x_280_; lean_object* v___x_281_; lean_object* v___x_283_; 
v___x_276_ = ((lean_object*)(lp_Qq_instToExprMData__qq___lam__0___closed__0));
v___x_277_ = ((lean_object*)(lp_Qq_instToExprMData__qq___lam__0___closed__9));
v___x_278_ = l_Lean_Name_mkStr3(v___x_220_, v___x_276_, v___x_277_);
v___x_279_ = l_Lean_Expr_const___override(v___x_278_, v___x_221_);
v___x_280_ = l_Lean_mkNatLit(v_v_272_);
v___x_281_ = l_Lean_mkApp3(v___x_279_, v___y_224_, v___x_230_, v___x_280_);
if (v_isShared_275_ == 0)
{
lean_ctor_set_tag(v___x_274_, 1);
lean_ctor_set(v___x_274_, 0, v___x_281_);
v___x_283_ = v___x_274_;
goto v_reusejp_282_;
}
else
{
lean_object* v_reuseFailAlloc_284_; 
v_reuseFailAlloc_284_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_284_, 0, v___x_281_);
v___x_283_ = v_reuseFailAlloc_284_;
goto v_reusejp_282_;
}
v_reusejp_282_:
{
return v___x_283_;
}
}
}
case 4:
{
lean_object* v_v_286_; lean_object* v___x_288_; uint8_t v_isShared_289_; uint8_t v_isSharedCheck_320_; 
v_v_286_ = lean_ctor_get(v_snd_226_, 0);
v_isSharedCheck_320_ = !lean_is_exclusive(v_snd_226_);
if (v_isSharedCheck_320_ == 0)
{
v___x_288_ = v_snd_226_;
v_isShared_289_ = v_isSharedCheck_320_;
goto v_resetjp_287_;
}
else
{
lean_inc(v_v_286_);
lean_dec(v_snd_226_);
v___x_288_ = lean_box(0);
v_isShared_289_ = v_isSharedCheck_320_;
goto v_resetjp_287_;
}
v_resetjp_287_:
{
lean_object* v___x_290_; lean_object* v___x_291_; lean_object* v___x_292_; lean_object* v___x_293_; lean_object* v___x_294_; uint8_t v___x_295_; 
v___x_290_ = ((lean_object*)(lp_Qq_instToExprMData__qq___lam__0___closed__0));
v___x_291_ = ((lean_object*)(lp_Qq_instToExprMData__qq___lam__0___closed__10));
v___x_292_ = l_Lean_Name_mkStr3(v___x_220_, v___x_290_, v___x_291_);
lean_inc(v___x_221_);
v___x_293_ = l_Lean_Expr_const___override(v___x_292_, v___x_221_);
v___x_294_ = lean_obj_once(&lp_Qq_instToExprMData__qq___lam__0___closed__11, &lp_Qq_instToExprMData__qq___lam__0___closed__11_once, _init_lp_Qq_instToExprMData__qq___lam__0___closed__11);
v___x_295_ = lean_int_dec_le(v___x_294_, v_v_286_);
if (v___x_295_ == 0)
{
lean_object* v___x_296_; lean_object* v___x_297_; lean_object* v___x_299_; 
v___x_296_ = ((lean_object*)(lp_Qq_instToExprMData__qq___lam__0___closed__14));
v___x_297_ = lean_obj_once(&lp_Qq_instToExprMData__qq___lam__0___closed__15, &lp_Qq_instToExprMData__qq___lam__0___closed__15_once, _init_lp_Qq_instToExprMData__qq___lam__0___closed__15);
lean_inc(v___x_221_);
if (v_isShared_229_ == 0)
{
lean_ctor_set_tag(v___x_228_, 1);
lean_ctor_set(v___x_228_, 1, v___x_221_);
lean_ctor_set(v___x_228_, 0, v___x_297_);
v___x_299_ = v___x_228_;
goto v_reusejp_298_;
}
else
{
lean_object* v_reuseFailAlloc_313_; 
v_reuseFailAlloc_313_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_313_, 0, v___x_297_);
lean_ctor_set(v_reuseFailAlloc_313_, 1, v___x_221_);
v___x_299_ = v_reuseFailAlloc_313_;
goto v_reusejp_298_;
}
v_reusejp_298_:
{
lean_object* v___x_300_; lean_object* v___x_301_; lean_object* v___x_302_; lean_object* v___x_303_; lean_object* v___x_304_; lean_object* v___x_305_; lean_object* v___x_306_; lean_object* v___x_307_; lean_object* v___x_308_; lean_object* v___x_309_; lean_object* v___x_311_; 
v___x_300_ = l_Lean_Expr_const___override(v___x_296_, v___x_299_);
v___x_301_ = ((lean_object*)(lp_Qq_instToExprMData__qq___lam__0___closed__17));
lean_inc(v___x_221_);
v___x_302_ = l_Lean_Expr_const___override(v___x_301_, v___x_221_);
v___x_303_ = ((lean_object*)(lp_Qq_instToExprMData__qq___lam__0___closed__19));
v___x_304_ = l_Lean_Expr_const___override(v___x_303_, v___x_221_);
v___x_305_ = lean_int_neg(v_v_286_);
lean_dec(v_v_286_);
v___x_306_ = l_Int_toNat(v___x_305_);
lean_dec(v___x_305_);
v___x_307_ = l_Lean_instToExprInt_mkNat(v___x_306_);
v___x_308_ = l_Lean_mkApp3(v___x_300_, v___x_302_, v___x_304_, v___x_307_);
v___x_309_ = l_Lean_mkApp3(v___x_293_, v___y_224_, v___x_230_, v___x_308_);
if (v_isShared_289_ == 0)
{
lean_ctor_set_tag(v___x_288_, 1);
lean_ctor_set(v___x_288_, 0, v___x_309_);
v___x_311_ = v___x_288_;
goto v_reusejp_310_;
}
else
{
lean_object* v_reuseFailAlloc_312_; 
v_reuseFailAlloc_312_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_312_, 0, v___x_309_);
v___x_311_ = v_reuseFailAlloc_312_;
goto v_reusejp_310_;
}
v_reusejp_310_:
{
return v___x_311_;
}
}
}
else
{
lean_object* v___x_314_; lean_object* v___x_315_; lean_object* v___x_316_; lean_object* v___x_318_; 
lean_del_object(v___x_228_);
lean_dec(v___x_221_);
v___x_314_ = l_Int_toNat(v_v_286_);
lean_dec(v_v_286_);
v___x_315_ = l_Lean_instToExprInt_mkNat(v___x_314_);
v___x_316_ = l_Lean_mkApp3(v___x_293_, v___y_224_, v___x_230_, v___x_315_);
if (v_isShared_289_ == 0)
{
lean_ctor_set_tag(v___x_288_, 1);
lean_ctor_set(v___x_288_, 0, v___x_316_);
v___x_318_ = v___x_288_;
goto v_reusejp_317_;
}
else
{
lean_object* v_reuseFailAlloc_319_; 
v_reuseFailAlloc_319_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_319_, 0, v___x_316_);
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
default: 
{
lean_object* v___x_322_; uint8_t v_isShared_323_; uint8_t v_isSharedCheck_327_; 
lean_dec_ref(v___x_230_);
lean_del_object(v___x_228_);
lean_dec(v___x_221_);
lean_dec_ref(v___x_220_);
v_isSharedCheck_327_ = !lean_is_exclusive(v_snd_226_);
if (v_isSharedCheck_327_ == 0)
{
lean_object* v_unused_328_; 
v_unused_328_ = lean_ctor_get(v_snd_226_, 0);
lean_dec(v_unused_328_);
v___x_322_ = v_snd_226_;
v_isShared_323_ = v_isSharedCheck_327_;
goto v_resetjp_321_;
}
else
{
lean_dec(v_snd_226_);
v___x_322_ = lean_box(0);
v_isShared_323_ = v_isSharedCheck_327_;
goto v_resetjp_321_;
}
v_resetjp_321_:
{
lean_object* v___x_325_; 
if (v_isShared_323_ == 0)
{
lean_ctor_set_tag(v___x_322_, 1);
lean_ctor_set(v___x_322_, 0, v___y_224_);
v___x_325_ = v___x_322_;
goto v_reusejp_324_;
}
else
{
lean_object* v_reuseFailAlloc_326_; 
v_reuseFailAlloc_326_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_326_, 0, v___y_224_);
v___x_325_ = v_reuseFailAlloc_326_;
goto v_reusejp_324_;
}
v_reusejp_324_:
{
return v___x_325_;
}
}
}
}
}
}
}
static lean_object* _init_lp_Qq_instToExprMData__qq___lam__1___closed__4(void){
_start:
{
lean_object* v___x_339_; lean_object* v___x_340_; lean_object* v_e_341_; 
v___x_339_ = lean_box(0);
v___x_340_ = ((lean_object*)(lp_Qq_instToExprMData__qq___lam__1___closed__2));
v_e_341_ = l_Lean_Expr_const___override(v___x_340_, v___x_339_);
return v_e_341_;
}
}
LEAN_EXPORT lean_object* lp_Qq_instToExprMData__qq___lam__1(lean_object* v___x_342_, lean_object* v_md_343_){
_start:
{
lean_object* v___f_344_; lean_object* v_e_345_; lean_object* v___x_346_; 
v___f_344_ = ((lean_object*)(lp_Qq_instToExprMData__qq___lam__1___closed__3));
v_e_345_ = lean_obj_once(&lp_Qq_instToExprMData__qq___lam__1___closed__4, &lp_Qq_instToExprMData__qq___lam__1___closed__4_once, _init_lp_Qq_instToExprMData__qq___lam__1___closed__4);
v___x_346_ = l_List_forIn_x27_loop___redArg(v___x_342_, v___f_344_, v_md_343_, v_e_345_);
return v___x_346_;
}
}
LEAN_EXPORT lean_object* lp_Qq_instToExprMData__qq___lam__1___boxed(lean_object* v___x_347_, lean_object* v_md_348_){
_start:
{
lean_object* v_res_349_; 
v_res_349_ = lp_Qq_instToExprMData__qq___lam__1(v___x_347_, v_md_348_);
lean_dec(v_md_348_);
return v_res_349_;
}
}
static lean_object* _init_lp_Qq_instToExprMData__qq___closed__12(void){
_start:
{
lean_object* v___x_374_; lean_object* v___x_375_; lean_object* v___x_376_; 
v___x_374_ = lean_box(0);
v___x_375_ = ((lean_object*)(lp_Qq_instToExprMData__qq___closed__11));
v___x_376_ = l_Lean_Expr_const___override(v___x_375_, v___x_374_);
return v___x_376_;
}
}
static lean_object* _init_lp_Qq_instToExprMData__qq___closed__13(void){
_start:
{
lean_object* v___x_377_; lean_object* v___f_378_; lean_object* v___x_379_; 
v___x_377_ = lean_obj_once(&lp_Qq_instToExprMData__qq___closed__12, &lp_Qq_instToExprMData__qq___closed__12_once, _init_lp_Qq_instToExprMData__qq___closed__12);
v___f_378_ = ((lean_object*)(lp_Qq_instToExprMData__qq___closed__10));
v___x_379_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_379_, 0, v___f_378_);
lean_ctor_set(v___x_379_, 1, v___x_377_);
return v___x_379_;
}
}
static lean_object* _init_lp_Qq_instToExprMData__qq(void){
_start:
{
lean_object* v___x_380_; 
v___x_380_ = lean_obj_once(&lp_Qq_instToExprMData__qq___closed__13, &lp_Qq_instToExprMData__qq___closed__13_once, _init_lp_Qq_instToExprMData__qq___closed__13);
return v___x_380_;
}
}
static lean_object* _init_lp_Qq_List_forIn_x27_loop___at___00toExprExpr_spec__1___redArg___closed__1(void){
_start:
{
lean_object* v___x_385_; lean_object* v___x_386_; lean_object* v___x_387_; 
v___x_385_ = lean_box(0);
v___x_386_ = ((lean_object*)(lp_Qq_List_forIn_x27_loop___at___00toExprExpr_spec__1___redArg___closed__0));
v___x_387_ = l_Lean_Expr_const___override(v___x_386_, v___x_385_);
return v___x_387_;
}
}
static lean_object* _init_lp_Qq_List_forIn_x27_loop___at___00toExprExpr_spec__1___redArg___closed__3(void){
_start:
{
lean_object* v___x_392_; lean_object* v___x_393_; lean_object* v___x_394_; 
v___x_392_ = lean_box(0);
v___x_393_ = ((lean_object*)(lp_Qq_List_forIn_x27_loop___at___00toExprExpr_spec__1___redArg___closed__2));
v___x_394_ = l_Lean_Expr_const___override(v___x_393_, v___x_392_);
return v___x_394_;
}
}
static lean_object* _init_lp_Qq_List_forIn_x27_loop___at___00toExprExpr_spec__1___redArg___closed__4(void){
_start:
{
lean_object* v___x_395_; lean_object* v___x_396_; lean_object* v___x_397_; 
v___x_395_ = lean_box(0);
v___x_396_ = ((lean_object*)(lp_Qq_instToExprMData__qq___lam__0___closed__5));
v___x_397_ = l_Lean_mkConst(v___x_396_, v___x_395_);
return v___x_397_;
}
}
static lean_object* _init_lp_Qq_List_forIn_x27_loop___at___00toExprExpr_spec__1___redArg___closed__5(void){
_start:
{
lean_object* v___x_398_; lean_object* v___x_399_; lean_object* v___x_400_; 
v___x_398_ = lean_box(0);
v___x_399_ = ((lean_object*)(lp_Qq_instToExprMData__qq___lam__0___closed__7));
v___x_400_ = l_Lean_mkConst(v___x_399_, v___x_398_);
return v___x_400_;
}
}
static lean_object* _init_lp_Qq_List_forIn_x27_loop___at___00toExprExpr_spec__1___redArg___closed__7(void){
_start:
{
lean_object* v___x_405_; lean_object* v___x_406_; lean_object* v___x_407_; 
v___x_405_ = lean_box(0);
v___x_406_ = ((lean_object*)(lp_Qq_List_forIn_x27_loop___at___00toExprExpr_spec__1___redArg___closed__6));
v___x_407_ = l_Lean_Expr_const___override(v___x_406_, v___x_405_);
return v___x_407_;
}
}
static lean_object* _init_lp_Qq_List_forIn_x27_loop___at___00toExprExpr_spec__1___redArg___closed__9(void){
_start:
{
lean_object* v___x_412_; lean_object* v___x_413_; lean_object* v___x_414_; 
v___x_412_ = lean_box(0);
v___x_413_ = ((lean_object*)(lp_Qq_List_forIn_x27_loop___at___00toExprExpr_spec__1___redArg___closed__8));
v___x_414_ = l_Lean_Expr_const___override(v___x_413_, v___x_412_);
return v___x_414_;
}
}
static lean_object* _init_lp_Qq_List_forIn_x27_loop___at___00toExprExpr_spec__1___redArg___closed__11(void){
_start:
{
lean_object* v___x_419_; lean_object* v___x_420_; lean_object* v___x_421_; 
v___x_419_ = lean_box(0);
v___x_420_ = ((lean_object*)(lp_Qq_List_forIn_x27_loop___at___00toExprExpr_spec__1___redArg___closed__10));
v___x_421_ = l_Lean_Expr_const___override(v___x_420_, v___x_419_);
return v___x_421_;
}
}
static lean_object* _init_lp_Qq_List_forIn_x27_loop___at___00toExprExpr_spec__1___redArg___closed__12(void){
_start:
{
lean_object* v___x_422_; lean_object* v___x_423_; lean_object* v___x_424_; 
v___x_422_ = lean_box(0);
v___x_423_ = lean_obj_once(&lp_Qq_instToExprMData__qq___lam__0___closed__15, &lp_Qq_instToExprMData__qq___lam__0___closed__15_once, _init_lp_Qq_instToExprMData__qq___lam__0___closed__15);
v___x_424_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_424_, 0, v___x_423_);
lean_ctor_set(v___x_424_, 1, v___x_422_);
return v___x_424_;
}
}
static lean_object* _init_lp_Qq_List_forIn_x27_loop___at___00toExprExpr_spec__1___redArg___closed__13(void){
_start:
{
lean_object* v___x_425_; lean_object* v___x_426_; lean_object* v___x_427_; 
v___x_425_ = lean_obj_once(&lp_Qq_List_forIn_x27_loop___at___00toExprExpr_spec__1___redArg___closed__12, &lp_Qq_List_forIn_x27_loop___at___00toExprExpr_spec__1___redArg___closed__12_once, _init_lp_Qq_List_forIn_x27_loop___at___00toExprExpr_spec__1___redArg___closed__12);
v___x_426_ = ((lean_object*)(lp_Qq_instToExprMData__qq___lam__0___closed__14));
v___x_427_ = l_Lean_Expr_const___override(v___x_426_, v___x_425_);
return v___x_427_;
}
}
static lean_object* _init_lp_Qq_List_forIn_x27_loop___at___00toExprExpr_spec__1___redArg___closed__14(void){
_start:
{
lean_object* v___x_428_; lean_object* v___x_429_; lean_object* v___x_430_; 
v___x_428_ = lean_box(0);
v___x_429_ = ((lean_object*)(lp_Qq_instToExprMData__qq___lam__0___closed__17));
v___x_430_ = l_Lean_Expr_const___override(v___x_429_, v___x_428_);
return v___x_430_;
}
}
static lean_object* _init_lp_Qq_List_forIn_x27_loop___at___00toExprExpr_spec__1___redArg___closed__15(void){
_start:
{
lean_object* v___x_431_; lean_object* v___x_432_; lean_object* v___x_433_; 
v___x_431_ = lean_box(0);
v___x_432_ = ((lean_object*)(lp_Qq_instToExprMData__qq___lam__0___closed__19));
v___x_433_ = l_Lean_Expr_const___override(v___x_432_, v___x_431_);
return v___x_433_;
}
}
LEAN_EXPORT lean_object* lp_Qq_List_forIn_x27_loop___at___00toExprExpr_spec__1___redArg(lean_object* v_as_x27_434_, lean_object* v_b_435_){
_start:
{
if (lean_obj_tag(v_as_x27_434_) == 0)
{
return v_b_435_;
}
else
{
lean_object* v_head_436_; lean_object* v_tail_437_; lean_object* v_fst_438_; lean_object* v_snd_439_; lean_object* v___x_440_; 
v_head_436_ = lean_ctor_get(v_as_x27_434_, 0);
v_tail_437_ = lean_ctor_get(v_as_x27_434_, 1);
v_fst_438_ = lean_ctor_get(v_head_436_, 0);
v_snd_439_ = lean_ctor_get(v_head_436_, 1);
lean_inc(v_fst_438_);
v___x_440_ = l___private_Lean_ToExpr_0__Lean_Name_toExprAux(v_fst_438_);
switch(lean_obj_tag(v_snd_439_))
{
case 0:
{
lean_object* v_v_441_; lean_object* v___x_442_; lean_object* v___x_443_; lean_object* v___x_444_; 
v_v_441_ = lean_ctor_get(v_snd_439_, 0);
v___x_442_ = lean_obj_once(&lp_Qq_List_forIn_x27_loop___at___00toExprExpr_spec__1___redArg___closed__1, &lp_Qq_List_forIn_x27_loop___at___00toExprExpr_spec__1___redArg___closed__1_once, _init_lp_Qq_List_forIn_x27_loop___at___00toExprExpr_spec__1___redArg___closed__1);
lean_inc_ref(v_v_441_);
v___x_443_ = l_Lean_mkStrLit(v_v_441_);
v___x_444_ = l_Lean_mkApp3(v___x_442_, v_b_435_, v___x_440_, v___x_443_);
v_as_x27_434_ = v_tail_437_;
v_b_435_ = v___x_444_;
goto _start;
}
case 1:
{
uint8_t v_v_446_; lean_object* v___x_447_; 
v_v_446_ = lean_ctor_get_uint8(v_snd_439_, 0);
v___x_447_ = lean_obj_once(&lp_Qq_List_forIn_x27_loop___at___00toExprExpr_spec__1___redArg___closed__3, &lp_Qq_List_forIn_x27_loop___at___00toExprExpr_spec__1___redArg___closed__3_once, _init_lp_Qq_List_forIn_x27_loop___at___00toExprExpr_spec__1___redArg___closed__3);
if (v_v_446_ == 0)
{
lean_object* v___x_448_; lean_object* v___x_449_; 
v___x_448_ = lean_obj_once(&lp_Qq_List_forIn_x27_loop___at___00toExprExpr_spec__1___redArg___closed__4, &lp_Qq_List_forIn_x27_loop___at___00toExprExpr_spec__1___redArg___closed__4_once, _init_lp_Qq_List_forIn_x27_loop___at___00toExprExpr_spec__1___redArg___closed__4);
v___x_449_ = l_Lean_mkApp3(v___x_447_, v_b_435_, v___x_440_, v___x_448_);
v_as_x27_434_ = v_tail_437_;
v_b_435_ = v___x_449_;
goto _start;
}
else
{
lean_object* v___x_451_; lean_object* v___x_452_; 
v___x_451_ = lean_obj_once(&lp_Qq_List_forIn_x27_loop___at___00toExprExpr_spec__1___redArg___closed__5, &lp_Qq_List_forIn_x27_loop___at___00toExprExpr_spec__1___redArg___closed__5_once, _init_lp_Qq_List_forIn_x27_loop___at___00toExprExpr_spec__1___redArg___closed__5);
v___x_452_ = l_Lean_mkApp3(v___x_447_, v_b_435_, v___x_440_, v___x_451_);
v_as_x27_434_ = v_tail_437_;
v_b_435_ = v___x_452_;
goto _start;
}
}
case 2:
{
lean_object* v_v_454_; lean_object* v___x_455_; lean_object* v___x_456_; lean_object* v___x_457_; 
v_v_454_ = lean_ctor_get(v_snd_439_, 0);
v___x_455_ = lean_obj_once(&lp_Qq_List_forIn_x27_loop___at___00toExprExpr_spec__1___redArg___closed__7, &lp_Qq_List_forIn_x27_loop___at___00toExprExpr_spec__1___redArg___closed__7_once, _init_lp_Qq_List_forIn_x27_loop___at___00toExprExpr_spec__1___redArg___closed__7);
lean_inc(v_v_454_);
v___x_456_ = l___private_Lean_ToExpr_0__Lean_Name_toExprAux(v_v_454_);
v___x_457_ = l_Lean_mkApp3(v___x_455_, v_b_435_, v___x_440_, v___x_456_);
v_as_x27_434_ = v_tail_437_;
v_b_435_ = v___x_457_;
goto _start;
}
case 3:
{
lean_object* v_v_459_; lean_object* v___x_460_; lean_object* v___x_461_; lean_object* v___x_462_; 
v_v_459_ = lean_ctor_get(v_snd_439_, 0);
v___x_460_ = lean_obj_once(&lp_Qq_List_forIn_x27_loop___at___00toExprExpr_spec__1___redArg___closed__9, &lp_Qq_List_forIn_x27_loop___at___00toExprExpr_spec__1___redArg___closed__9_once, _init_lp_Qq_List_forIn_x27_loop___at___00toExprExpr_spec__1___redArg___closed__9);
lean_inc(v_v_459_);
v___x_461_ = l_Lean_mkNatLit(v_v_459_);
v___x_462_ = l_Lean_mkApp3(v___x_460_, v_b_435_, v___x_440_, v___x_461_);
v_as_x27_434_ = v_tail_437_;
v_b_435_ = v___x_462_;
goto _start;
}
case 4:
{
lean_object* v_v_464_; lean_object* v___x_465_; lean_object* v___x_466_; uint8_t v___x_467_; 
v_v_464_ = lean_ctor_get(v_snd_439_, 0);
v___x_465_ = lean_obj_once(&lp_Qq_List_forIn_x27_loop___at___00toExprExpr_spec__1___redArg___closed__11, &lp_Qq_List_forIn_x27_loop___at___00toExprExpr_spec__1___redArg___closed__11_once, _init_lp_Qq_List_forIn_x27_loop___at___00toExprExpr_spec__1___redArg___closed__11);
v___x_466_ = lean_obj_once(&lp_Qq_instToExprMData__qq___lam__0___closed__11, &lp_Qq_instToExprMData__qq___lam__0___closed__11_once, _init_lp_Qq_instToExprMData__qq___lam__0___closed__11);
v___x_467_ = lean_int_dec_le(v___x_466_, v_v_464_);
if (v___x_467_ == 0)
{
lean_object* v___x_468_; lean_object* v___x_469_; lean_object* v___x_470_; lean_object* v___x_471_; lean_object* v___x_472_; lean_object* v___x_473_; lean_object* v___x_474_; lean_object* v___x_475_; 
v___x_468_ = lean_obj_once(&lp_Qq_List_forIn_x27_loop___at___00toExprExpr_spec__1___redArg___closed__13, &lp_Qq_List_forIn_x27_loop___at___00toExprExpr_spec__1___redArg___closed__13_once, _init_lp_Qq_List_forIn_x27_loop___at___00toExprExpr_spec__1___redArg___closed__13);
v___x_469_ = lean_obj_once(&lp_Qq_List_forIn_x27_loop___at___00toExprExpr_spec__1___redArg___closed__14, &lp_Qq_List_forIn_x27_loop___at___00toExprExpr_spec__1___redArg___closed__14_once, _init_lp_Qq_List_forIn_x27_loop___at___00toExprExpr_spec__1___redArg___closed__14);
v___x_470_ = lean_obj_once(&lp_Qq_List_forIn_x27_loop___at___00toExprExpr_spec__1___redArg___closed__15, &lp_Qq_List_forIn_x27_loop___at___00toExprExpr_spec__1___redArg___closed__15_once, _init_lp_Qq_List_forIn_x27_loop___at___00toExprExpr_spec__1___redArg___closed__15);
v___x_471_ = lean_int_neg(v_v_464_);
v___x_472_ = l_Int_toNat(v___x_471_);
lean_dec(v___x_471_);
v___x_473_ = l_Lean_instToExprInt_mkNat(v___x_472_);
v___x_474_ = l_Lean_mkApp3(v___x_468_, v___x_469_, v___x_470_, v___x_473_);
v___x_475_ = l_Lean_mkApp3(v___x_465_, v_b_435_, v___x_440_, v___x_474_);
v_as_x27_434_ = v_tail_437_;
v_b_435_ = v___x_475_;
goto _start;
}
else
{
lean_object* v___x_477_; lean_object* v___x_478_; lean_object* v___x_479_; 
v___x_477_ = l_Int_toNat(v_v_464_);
v___x_478_ = l_Lean_instToExprInt_mkNat(v___x_477_);
v___x_479_ = l_Lean_mkApp3(v___x_465_, v_b_435_, v___x_440_, v___x_478_);
v_as_x27_434_ = v_tail_437_;
v_b_435_ = v___x_479_;
goto _start;
}
}
default: 
{
lean_dec_ref(v___x_440_);
v_as_x27_434_ = v_tail_437_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_Qq_List_forIn_x27_loop___at___00toExprExpr_spec__1___redArg___boxed(lean_object* v_as_x27_482_, lean_object* v_b_483_){
_start:
{
lean_object* v_res_484_; 
v_res_484_ = lp_Qq_List_forIn_x27_loop___at___00toExprExpr_spec__1___redArg(v_as_x27_482_, v_b_483_);
lean_dec(v_as_x27_482_);
return v_res_484_;
}
}
LEAN_EXPORT lean_object* lp_Qq___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00toExprExpr_spec__0(lean_object* v_nilFn_485_, lean_object* v_consFn_486_, lean_object* v_x_487_){
_start:
{
if (lean_obj_tag(v_x_487_) == 0)
{
lean_dec_ref(v_consFn_486_);
lean_inc_ref(v_nilFn_485_);
return v_nilFn_485_;
}
else
{
lean_object* v_head_488_; lean_object* v_tail_489_; lean_object* v___x_490_; lean_object* v___x_491_; lean_object* v___x_492_; 
v_head_488_ = lean_ctor_get(v_x_487_, 0);
lean_inc(v_head_488_);
v_tail_489_ = lean_ctor_get(v_x_487_, 1);
lean_inc(v_tail_489_);
lean_dec_ref_known(v_x_487_, 2);
v___x_490_ = lp_Qq_toExprLevel(v_head_488_);
lean_inc_ref(v_consFn_486_);
v___x_491_ = lp_Qq___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00toExprExpr_spec__0(v_nilFn_485_, v_consFn_486_, v_tail_489_);
v___x_492_ = l_Lean_mkAppB(v_consFn_486_, v___x_490_, v___x_491_);
return v___x_492_;
}
}
}
LEAN_EXPORT lean_object* lp_Qq___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00toExprExpr_spec__0___boxed(lean_object* v_nilFn_493_, lean_object* v_consFn_494_, lean_object* v_x_495_){
_start:
{
lean_object* v_res_496_; 
v_res_496_ = lp_Qq___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00toExprExpr_spec__0(v_nilFn_493_, v_consFn_494_, v_x_495_);
lean_dec_ref(v_nilFn_493_);
return v_res_496_;
}
}
static lean_object* _init_lp_Qq_toExprExpr___closed__3(void){
_start:
{
lean_object* v___x_503_; lean_object* v___x_504_; lean_object* v___x_505_; 
v___x_503_ = lean_box(0);
v___x_504_ = ((lean_object*)(lp_Qq_toExprExpr___closed__2));
v___x_505_ = l_Lean_Expr_const___override(v___x_504_, v___x_503_);
return v___x_505_;
}
}
static lean_object* _init_lp_Qq_toExprExpr___closed__6(void){
_start:
{
lean_object* v___x_511_; lean_object* v___x_512_; lean_object* v___x_513_; 
v___x_511_ = lean_box(0);
v___x_512_ = ((lean_object*)(lp_Qq_toExprExpr___closed__5));
v___x_513_ = l_Lean_Expr_const___override(v___x_512_, v___x_511_);
return v___x_513_;
}
}
static lean_object* _init_lp_Qq_toExprExpr___closed__9(void){
_start:
{
lean_object* v___x_519_; lean_object* v___x_520_; lean_object* v___x_521_; 
v___x_519_ = lean_box(0);
v___x_520_ = ((lean_object*)(lp_Qq_toExprExpr___closed__8));
v___x_521_ = l_Lean_mkConst(v___x_520_, v___x_519_);
return v___x_521_;
}
}
static lean_object* _init_lp_Qq_toExprExpr___closed__11(void){
_start:
{
lean_object* v___x_526_; lean_object* v___x_527_; lean_object* v___x_528_; 
v___x_526_ = lean_box(0);
v___x_527_ = ((lean_object*)(lp_Qq_toExprExpr___closed__10));
v___x_528_ = l_Lean_Expr_const___override(v___x_527_, v___x_526_);
return v___x_528_;
}
}
static lean_object* _init_lp_Qq_toExprExpr___closed__14(void){
_start:
{
lean_object* v___x_534_; lean_object* v___x_535_; lean_object* v___x_536_; 
v___x_534_ = lean_box(0);
v___x_535_ = ((lean_object*)(lp_Qq_toExprExpr___closed__13));
v___x_536_ = l_Lean_Expr_const___override(v___x_535_, v___x_534_);
return v___x_536_;
}
}
static lean_object* _init_lp_Qq_toExprExpr___closed__17(void){
_start:
{
lean_object* v___x_542_; lean_object* v___x_543_; lean_object* v___x_544_; 
v___x_542_ = lean_box(0);
v___x_543_ = ((lean_object*)(lp_Qq_toExprExpr___closed__16));
v___x_544_ = l_Lean_Expr_const___override(v___x_543_, v___x_542_);
return v___x_544_;
}
}
static lean_object* _init_lp_Qq_toExprExpr___closed__22(void){
_start:
{
lean_object* v___x_553_; lean_object* v___x_554_; lean_object* v___x_555_; 
v___x_553_ = ((lean_object*)(lp_Qq_toExprExpr___closed__21));
v___x_554_ = ((lean_object*)(lp_Qq_toExprExpr___closed__20));
v___x_555_ = l_Lean_mkConst(v___x_554_, v___x_553_);
return v___x_555_;
}
}
static lean_object* _init_lp_Qq_toExprExpr___closed__23(void){
_start:
{
lean_object* v_type_556_; lean_object* v___x_557_; lean_object* v_nil_558_; 
v_type_556_ = lean_obj_once(&lp_Qq_instToExprLevel__qq___closed__2, &lp_Qq_instToExprLevel__qq___closed__2_once, _init_lp_Qq_instToExprLevel__qq___closed__2);
v___x_557_ = lean_obj_once(&lp_Qq_toExprExpr___closed__22, &lp_Qq_toExprExpr___closed__22_once, _init_lp_Qq_toExprExpr___closed__22);
v_nil_558_ = l_Lean_Expr_app___override(v___x_557_, v_type_556_);
return v_nil_558_;
}
}
static lean_object* _init_lp_Qq_toExprExpr___closed__26(void){
_start:
{
lean_object* v___x_563_; lean_object* v___x_564_; lean_object* v___x_565_; 
v___x_563_ = ((lean_object*)(lp_Qq_toExprExpr___closed__21));
v___x_564_ = ((lean_object*)(lp_Qq_toExprExpr___closed__25));
v___x_565_ = l_Lean_mkConst(v___x_564_, v___x_563_);
return v___x_565_;
}
}
static lean_object* _init_lp_Qq_toExprExpr___closed__27(void){
_start:
{
lean_object* v_type_566_; lean_object* v___x_567_; lean_object* v_cons_568_; 
v_type_566_ = lean_obj_once(&lp_Qq_instToExprLevel__qq___closed__2, &lp_Qq_instToExprLevel__qq___closed__2_once, _init_lp_Qq_instToExprLevel__qq___closed__2);
v___x_567_ = lean_obj_once(&lp_Qq_toExprExpr___closed__26, &lp_Qq_toExprExpr___closed__26_once, _init_lp_Qq_toExprExpr___closed__26);
v_cons_568_ = l_Lean_Expr_app___override(v___x_567_, v_type_566_);
return v_cons_568_;
}
}
static lean_object* _init_lp_Qq_toExprExpr___closed__30(void){
_start:
{
lean_object* v___x_574_; lean_object* v___x_575_; lean_object* v___x_576_; 
v___x_574_ = lean_box(0);
v___x_575_ = ((lean_object*)(lp_Qq_toExprExpr___closed__29));
v___x_576_ = l_Lean_Expr_const___override(v___x_575_, v___x_574_);
return v___x_576_;
}
}
static lean_object* _init_lp_Qq_toExprExpr___closed__33(void){
_start:
{
lean_object* v___x_582_; lean_object* v___x_583_; lean_object* v___x_584_; 
v___x_582_ = lean_box(0);
v___x_583_ = ((lean_object*)(lp_Qq_toExprExpr___closed__32));
v___x_584_ = l_Lean_Expr_const___override(v___x_583_, v___x_582_);
return v___x_584_;
}
}
static lean_object* _init_lp_Qq_toExprExpr___closed__36(void){
_start:
{
lean_object* v___x_590_; lean_object* v___x_591_; lean_object* v___x_592_; 
v___x_590_ = lean_box(0);
v___x_591_ = ((lean_object*)(lp_Qq_toExprExpr___closed__35));
v___x_592_ = l_Lean_Expr_const___override(v___x_591_, v___x_590_);
return v___x_592_;
}
}
static lean_object* _init_lp_Qq_toExprExpr___closed__39(void){
_start:
{
lean_object* v___x_598_; lean_object* v___x_599_; lean_object* v___x_600_; 
v___x_598_ = lean_box(0);
v___x_599_ = ((lean_object*)(lp_Qq_toExprExpr___closed__38));
v___x_600_ = l_Lean_Expr_const___override(v___x_599_, v___x_598_);
return v___x_600_;
}
}
static lean_object* _init_lp_Qq_toExprExpr___closed__42(void){
_start:
{
lean_object* v___x_606_; lean_object* v___x_607_; lean_object* v___x_608_; 
v___x_606_ = lean_box(0);
v___x_607_ = ((lean_object*)(lp_Qq_toExprExpr___closed__41));
v___x_608_ = l_Lean_Expr_const___override(v___x_607_, v___x_606_);
return v___x_608_;
}
}
static lean_object* _init_lp_Qq_toExprExpr___closed__46(void){
_start:
{
lean_object* v___x_615_; lean_object* v___x_616_; lean_object* v___x_617_; 
v___x_615_ = lean_box(0);
v___x_616_ = ((lean_object*)(lp_Qq_toExprExpr___closed__45));
v___x_617_ = l_Lean_mkConst(v___x_616_, v___x_615_);
return v___x_617_;
}
}
static lean_object* _init_lp_Qq_toExprExpr___closed__49(void){
_start:
{
lean_object* v___x_623_; lean_object* v___x_624_; lean_object* v___x_625_; 
v___x_623_ = lean_box(0);
v___x_624_ = ((lean_object*)(lp_Qq_toExprExpr___closed__48));
v___x_625_ = l_Lean_mkConst(v___x_624_, v___x_623_);
return v___x_625_;
}
}
static lean_object* _init_lp_Qq_toExprExpr___closed__52(void){
_start:
{
lean_object* v___x_631_; lean_object* v___x_632_; lean_object* v___x_633_; 
v___x_631_ = lean_box(0);
v___x_632_ = ((lean_object*)(lp_Qq_toExprExpr___closed__51));
v___x_633_ = l_Lean_Expr_const___override(v___x_632_, v___x_631_);
return v___x_633_;
}
}
static lean_object* _init_lp_Qq_toExprExpr___closed__55(void){
_start:
{
lean_object* v___x_639_; lean_object* v___x_640_; lean_object* v___x_641_; 
v___x_639_ = lean_box(0);
v___x_640_ = ((lean_object*)(lp_Qq_toExprExpr___closed__54));
v___x_641_ = l_Lean_Expr_const___override(v___x_640_, v___x_639_);
return v___x_641_;
}
}
LEAN_EXPORT lean_object* lp_Qq_toExprExpr(lean_object* v_x_642_){
_start:
{
switch(lean_obj_tag(v_x_642_))
{
case 0:
{
lean_object* v_deBruijnIndex_643_; lean_object* v___x_644_; lean_object* v___x_645_; lean_object* v___x_646_; 
v_deBruijnIndex_643_ = lean_ctor_get(v_x_642_, 0);
lean_inc(v_deBruijnIndex_643_);
lean_dec_ref_known(v_x_642_, 1);
v___x_644_ = lean_obj_once(&lp_Qq_toExprExpr___closed__3, &lp_Qq_toExprExpr___closed__3_once, _init_lp_Qq_toExprExpr___closed__3);
v___x_645_ = l_Lean_mkNatLit(v_deBruijnIndex_643_);
v___x_646_ = l_Lean_Expr_app___override(v___x_644_, v___x_645_);
return v___x_646_;
}
case 1:
{
lean_object* v_fvarId_647_; lean_object* v___x_648_; lean_object* v___x_649_; lean_object* v___x_650_; lean_object* v___x_651_; lean_object* v___x_652_; 
v_fvarId_647_ = lean_ctor_get(v_x_642_, 0);
lean_inc(v_fvarId_647_);
lean_dec_ref_known(v_x_642_, 1);
v___x_648_ = lean_obj_once(&lp_Qq_toExprExpr___closed__6, &lp_Qq_toExprExpr___closed__6_once, _init_lp_Qq_toExprExpr___closed__6);
v___x_649_ = lean_obj_once(&lp_Qq_toExprExpr___closed__9, &lp_Qq_toExprExpr___closed__9_once, _init_lp_Qq_toExprExpr___closed__9);
v___x_650_ = l___private_Lean_ToExpr_0__Lean_Name_toExprAux(v_fvarId_647_);
v___x_651_ = l_Lean_Expr_app___override(v___x_649_, v___x_650_);
v___x_652_ = l_Lean_Expr_app___override(v___x_648_, v___x_651_);
return v___x_652_;
}
case 2:
{
lean_object* v_mvarId_653_; lean_object* v___x_654_; lean_object* v___x_655_; lean_object* v___x_656_; lean_object* v___x_657_; lean_object* v___x_658_; 
v_mvarId_653_ = lean_ctor_get(v_x_642_, 0);
lean_inc(v_mvarId_653_);
lean_dec_ref_known(v_x_642_, 1);
v___x_654_ = lean_obj_once(&lp_Qq_toExprExpr___closed__11, &lp_Qq_toExprExpr___closed__11_once, _init_lp_Qq_toExprExpr___closed__11);
v___x_655_ = lean_obj_once(&lp_Qq_instToExprMVarId__qq___lam__0___closed__4, &lp_Qq_instToExprMVarId__qq___lam__0___closed__4_once, _init_lp_Qq_instToExprMVarId__qq___lam__0___closed__4);
v___x_656_ = l___private_Lean_ToExpr_0__Lean_Name_toExprAux(v_mvarId_653_);
v___x_657_ = l_Lean_Expr_app___override(v___x_655_, v___x_656_);
v___x_658_ = l_Lean_Expr_app___override(v___x_654_, v___x_657_);
return v___x_658_;
}
case 3:
{
lean_object* v_u_659_; lean_object* v___x_660_; lean_object* v___x_661_; lean_object* v___x_662_; 
v_u_659_ = lean_ctor_get(v_x_642_, 0);
lean_inc(v_u_659_);
lean_dec_ref_known(v_x_642_, 1);
v___x_660_ = lean_obj_once(&lp_Qq_toExprExpr___closed__14, &lp_Qq_toExprExpr___closed__14_once, _init_lp_Qq_toExprExpr___closed__14);
v___x_661_ = lp_Qq_toExprLevel(v_u_659_);
v___x_662_ = l_Lean_Expr_app___override(v___x_660_, v___x_661_);
return v___x_662_;
}
case 4:
{
lean_object* v_declName_663_; lean_object* v_us_664_; lean_object* v___x_665_; lean_object* v___x_666_; lean_object* v_nil_667_; lean_object* v_cons_668_; lean_object* v___x_669_; lean_object* v___x_670_; 
v_declName_663_ = lean_ctor_get(v_x_642_, 0);
lean_inc(v_declName_663_);
v_us_664_ = lean_ctor_get(v_x_642_, 1);
lean_inc(v_us_664_);
lean_dec_ref_known(v_x_642_, 2);
v___x_665_ = lean_obj_once(&lp_Qq_toExprExpr___closed__17, &lp_Qq_toExprExpr___closed__17_once, _init_lp_Qq_toExprExpr___closed__17);
v___x_666_ = l___private_Lean_ToExpr_0__Lean_Name_toExprAux(v_declName_663_);
v_nil_667_ = lean_obj_once(&lp_Qq_toExprExpr___closed__23, &lp_Qq_toExprExpr___closed__23_once, _init_lp_Qq_toExprExpr___closed__23);
v_cons_668_ = lean_obj_once(&lp_Qq_toExprExpr___closed__27, &lp_Qq_toExprExpr___closed__27_once, _init_lp_Qq_toExprExpr___closed__27);
v___x_669_ = lp_Qq___private_Lean_ToExpr_0__Lean_List_toExprAux___at___00toExprExpr_spec__0(v_nil_667_, v_cons_668_, v_us_664_);
v___x_670_ = l_Lean_mkAppB(v___x_665_, v___x_666_, v___x_669_);
return v___x_670_;
}
case 5:
{
lean_object* v_fn_671_; lean_object* v_arg_672_; lean_object* v___x_673_; lean_object* v___x_674_; lean_object* v___x_675_; lean_object* v___x_676_; 
v_fn_671_ = lean_ctor_get(v_x_642_, 0);
lean_inc_ref(v_fn_671_);
v_arg_672_ = lean_ctor_get(v_x_642_, 1);
lean_inc_ref(v_arg_672_);
lean_dec_ref_known(v_x_642_, 2);
v___x_673_ = lean_obj_once(&lp_Qq_toExprExpr___closed__30, &lp_Qq_toExprExpr___closed__30_once, _init_lp_Qq_toExprExpr___closed__30);
v___x_674_ = lp_Qq_toExprExpr(v_fn_671_);
v___x_675_ = lp_Qq_toExprExpr(v_arg_672_);
v___x_676_ = l_Lean_mkAppB(v___x_673_, v___x_674_, v___x_675_);
return v___x_676_;
}
case 6:
{
lean_object* v_binderName_677_; lean_object* v_binderType_678_; lean_object* v_body_679_; uint8_t v_binderInfo_680_; lean_object* v___x_681_; lean_object* v___x_682_; lean_object* v___x_683_; lean_object* v___x_684_; 
v_binderName_677_ = lean_ctor_get(v_x_642_, 0);
lean_inc(v_binderName_677_);
v_binderType_678_ = lean_ctor_get(v_x_642_, 1);
lean_inc_ref(v_binderType_678_);
v_body_679_ = lean_ctor_get(v_x_642_, 2);
lean_inc_ref(v_body_679_);
v_binderInfo_680_ = lean_ctor_get_uint8(v_x_642_, sizeof(void*)*3 + 8);
lean_dec_ref_known(v_x_642_, 3);
v___x_681_ = lean_obj_once(&lp_Qq_toExprExpr___closed__33, &lp_Qq_toExprExpr___closed__33_once, _init_lp_Qq_toExprExpr___closed__33);
v___x_682_ = l___private_Lean_ToExpr_0__Lean_Name_toExprAux(v_binderName_677_);
v___x_683_ = lp_Qq_toExprExpr(v_binderType_678_);
v___x_684_ = lp_Qq_toExprExpr(v_body_679_);
switch(v_binderInfo_680_)
{
case 0:
{
lean_object* v___x_685_; lean_object* v___x_686_; 
v___x_685_ = lean_obj_once(&lp_Qq_instToExprBinderInfo__qq___lam__0___closed__3, &lp_Qq_instToExprBinderInfo__qq___lam__0___closed__3_once, _init_lp_Qq_instToExprBinderInfo__qq___lam__0___closed__3);
v___x_686_ = l_Lean_mkApp4(v___x_681_, v___x_682_, v___x_683_, v___x_684_, v___x_685_);
return v___x_686_;
}
case 1:
{
lean_object* v___x_687_; lean_object* v___x_688_; 
v___x_687_ = lean_obj_once(&lp_Qq_instToExprBinderInfo__qq___lam__0___closed__6, &lp_Qq_instToExprBinderInfo__qq___lam__0___closed__6_once, _init_lp_Qq_instToExprBinderInfo__qq___lam__0___closed__6);
v___x_688_ = l_Lean_mkApp4(v___x_681_, v___x_682_, v___x_683_, v___x_684_, v___x_687_);
return v___x_688_;
}
case 2:
{
lean_object* v___x_689_; lean_object* v___x_690_; 
v___x_689_ = lean_obj_once(&lp_Qq_instToExprBinderInfo__qq___lam__0___closed__9, &lp_Qq_instToExprBinderInfo__qq___lam__0___closed__9_once, _init_lp_Qq_instToExprBinderInfo__qq___lam__0___closed__9);
v___x_690_ = l_Lean_mkApp4(v___x_681_, v___x_682_, v___x_683_, v___x_684_, v___x_689_);
return v___x_690_;
}
default: 
{
lean_object* v___x_691_; lean_object* v___x_692_; 
v___x_691_ = lean_obj_once(&lp_Qq_instToExprBinderInfo__qq___lam__0___closed__12, &lp_Qq_instToExprBinderInfo__qq___lam__0___closed__12_once, _init_lp_Qq_instToExprBinderInfo__qq___lam__0___closed__12);
v___x_692_ = l_Lean_mkApp4(v___x_681_, v___x_682_, v___x_683_, v___x_684_, v___x_691_);
return v___x_692_;
}
}
}
case 7:
{
lean_object* v_binderName_693_; lean_object* v_binderType_694_; lean_object* v_body_695_; uint8_t v_binderInfo_696_; lean_object* v___x_697_; lean_object* v___x_698_; lean_object* v___x_699_; lean_object* v___x_700_; 
v_binderName_693_ = lean_ctor_get(v_x_642_, 0);
lean_inc(v_binderName_693_);
v_binderType_694_ = lean_ctor_get(v_x_642_, 1);
lean_inc_ref(v_binderType_694_);
v_body_695_ = lean_ctor_get(v_x_642_, 2);
lean_inc_ref(v_body_695_);
v_binderInfo_696_ = lean_ctor_get_uint8(v_x_642_, sizeof(void*)*3 + 8);
lean_dec_ref_known(v_x_642_, 3);
v___x_697_ = lean_obj_once(&lp_Qq_toExprExpr___closed__36, &lp_Qq_toExprExpr___closed__36_once, _init_lp_Qq_toExprExpr___closed__36);
v___x_698_ = l___private_Lean_ToExpr_0__Lean_Name_toExprAux(v_binderName_693_);
v___x_699_ = lp_Qq_toExprExpr(v_binderType_694_);
v___x_700_ = lp_Qq_toExprExpr(v_body_695_);
switch(v_binderInfo_696_)
{
case 0:
{
lean_object* v___x_701_; lean_object* v___x_702_; 
v___x_701_ = lean_obj_once(&lp_Qq_instToExprBinderInfo__qq___lam__0___closed__3, &lp_Qq_instToExprBinderInfo__qq___lam__0___closed__3_once, _init_lp_Qq_instToExprBinderInfo__qq___lam__0___closed__3);
v___x_702_ = l_Lean_mkApp4(v___x_697_, v___x_698_, v___x_699_, v___x_700_, v___x_701_);
return v___x_702_;
}
case 1:
{
lean_object* v___x_703_; lean_object* v___x_704_; 
v___x_703_ = lean_obj_once(&lp_Qq_instToExprBinderInfo__qq___lam__0___closed__6, &lp_Qq_instToExprBinderInfo__qq___lam__0___closed__6_once, _init_lp_Qq_instToExprBinderInfo__qq___lam__0___closed__6);
v___x_704_ = l_Lean_mkApp4(v___x_697_, v___x_698_, v___x_699_, v___x_700_, v___x_703_);
return v___x_704_;
}
case 2:
{
lean_object* v___x_705_; lean_object* v___x_706_; 
v___x_705_ = lean_obj_once(&lp_Qq_instToExprBinderInfo__qq___lam__0___closed__9, &lp_Qq_instToExprBinderInfo__qq___lam__0___closed__9_once, _init_lp_Qq_instToExprBinderInfo__qq___lam__0___closed__9);
v___x_706_ = l_Lean_mkApp4(v___x_697_, v___x_698_, v___x_699_, v___x_700_, v___x_705_);
return v___x_706_;
}
default: 
{
lean_object* v___x_707_; lean_object* v___x_708_; 
v___x_707_ = lean_obj_once(&lp_Qq_instToExprBinderInfo__qq___lam__0___closed__12, &lp_Qq_instToExprBinderInfo__qq___lam__0___closed__12_once, _init_lp_Qq_instToExprBinderInfo__qq___lam__0___closed__12);
v___x_708_ = l_Lean_mkApp4(v___x_697_, v___x_698_, v___x_699_, v___x_700_, v___x_707_);
return v___x_708_;
}
}
}
case 8:
{
lean_object* v_declName_709_; lean_object* v_type_710_; lean_object* v_value_711_; lean_object* v_body_712_; uint8_t v_nondep_713_; lean_object* v___x_714_; lean_object* v___x_715_; lean_object* v___x_716_; lean_object* v___x_717_; lean_object* v___x_718_; 
v_declName_709_ = lean_ctor_get(v_x_642_, 0);
lean_inc(v_declName_709_);
v_type_710_ = lean_ctor_get(v_x_642_, 1);
lean_inc_ref(v_type_710_);
v_value_711_ = lean_ctor_get(v_x_642_, 2);
lean_inc_ref(v_value_711_);
v_body_712_ = lean_ctor_get(v_x_642_, 3);
lean_inc_ref(v_body_712_);
v_nondep_713_ = lean_ctor_get_uint8(v_x_642_, sizeof(void*)*4 + 8);
lean_dec_ref_known(v_x_642_, 4);
v___x_714_ = lean_obj_once(&lp_Qq_toExprExpr___closed__39, &lp_Qq_toExprExpr___closed__39_once, _init_lp_Qq_toExprExpr___closed__39);
v___x_715_ = l___private_Lean_ToExpr_0__Lean_Name_toExprAux(v_declName_709_);
v___x_716_ = lp_Qq_toExprExpr(v_type_710_);
v___x_717_ = lp_Qq_toExprExpr(v_value_711_);
v___x_718_ = lp_Qq_toExprExpr(v_body_712_);
if (v_nondep_713_ == 0)
{
lean_object* v___x_719_; lean_object* v___x_720_; 
v___x_719_ = lean_obj_once(&lp_Qq_List_forIn_x27_loop___at___00toExprExpr_spec__1___redArg___closed__4, &lp_Qq_List_forIn_x27_loop___at___00toExprExpr_spec__1___redArg___closed__4_once, _init_lp_Qq_List_forIn_x27_loop___at___00toExprExpr_spec__1___redArg___closed__4);
v___x_720_ = l_Lean_mkApp5(v___x_714_, v___x_715_, v___x_716_, v___x_717_, v___x_718_, v___x_719_);
return v___x_720_;
}
else
{
lean_object* v___x_721_; lean_object* v___x_722_; 
v___x_721_ = lean_obj_once(&lp_Qq_List_forIn_x27_loop___at___00toExprExpr_spec__1___redArg___closed__5, &lp_Qq_List_forIn_x27_loop___at___00toExprExpr_spec__1___redArg___closed__5_once, _init_lp_Qq_List_forIn_x27_loop___at___00toExprExpr_spec__1___redArg___closed__5);
v___x_722_ = l_Lean_mkApp5(v___x_714_, v___x_715_, v___x_716_, v___x_717_, v___x_718_, v___x_721_);
return v___x_722_;
}
}
case 9:
{
lean_object* v_a_723_; lean_object* v___x_724_; 
v_a_723_ = lean_ctor_get(v_x_642_, 0);
lean_inc_ref(v_a_723_);
lean_dec_ref_known(v_x_642_, 1);
v___x_724_ = lean_obj_once(&lp_Qq_toExprExpr___closed__42, &lp_Qq_toExprExpr___closed__42_once, _init_lp_Qq_toExprExpr___closed__42);
if (lean_obj_tag(v_a_723_) == 0)
{
lean_object* v___x_725_; lean_object* v___x_726_; lean_object* v___x_727_; lean_object* v___x_728_; 
v___x_725_ = lean_obj_once(&lp_Qq_toExprExpr___closed__46, &lp_Qq_toExprExpr___closed__46_once, _init_lp_Qq_toExprExpr___closed__46);
v___x_726_ = l_Lean_Expr_lit___override(v_a_723_);
v___x_727_ = l_Lean_Expr_app___override(v___x_725_, v___x_726_);
v___x_728_ = l_Lean_Expr_app___override(v___x_724_, v___x_727_);
return v___x_728_;
}
else
{
lean_object* v___x_729_; lean_object* v___x_730_; lean_object* v___x_731_; lean_object* v___x_732_; 
v___x_729_ = lean_obj_once(&lp_Qq_toExprExpr___closed__49, &lp_Qq_toExprExpr___closed__49_once, _init_lp_Qq_toExprExpr___closed__49);
v___x_730_ = l_Lean_Expr_lit___override(v_a_723_);
v___x_731_ = l_Lean_Expr_app___override(v___x_729_, v___x_730_);
v___x_732_ = l_Lean_Expr_app___override(v___x_724_, v___x_731_);
return v___x_732_;
}
}
case 10:
{
lean_object* v_data_733_; lean_object* v_expr_734_; lean_object* v___x_735_; lean_object* v_e_736_; lean_object* v___x_737_; lean_object* v___x_738_; lean_object* v___x_739_; 
v_data_733_ = lean_ctor_get(v_x_642_, 0);
lean_inc(v_data_733_);
v_expr_734_ = lean_ctor_get(v_x_642_, 1);
lean_inc_ref(v_expr_734_);
lean_dec_ref_known(v_x_642_, 2);
v___x_735_ = lean_obj_once(&lp_Qq_toExprExpr___closed__52, &lp_Qq_toExprExpr___closed__52_once, _init_lp_Qq_toExprExpr___closed__52);
v_e_736_ = lean_obj_once(&lp_Qq_instToExprMData__qq___lam__1___closed__4, &lp_Qq_instToExprMData__qq___lam__1___closed__4_once, _init_lp_Qq_instToExprMData__qq___lam__1___closed__4);
v___x_737_ = lp_Qq_List_forIn_x27_loop___at___00toExprExpr_spec__1___redArg(v_data_733_, v_e_736_);
lean_dec(v_data_733_);
v___x_738_ = lp_Qq_toExprExpr(v_expr_734_);
v___x_739_ = l_Lean_mkAppB(v___x_735_, v___x_737_, v___x_738_);
return v___x_739_;
}
default: 
{
lean_object* v_typeName_740_; lean_object* v_idx_741_; lean_object* v_struct_742_; lean_object* v___x_743_; lean_object* v___x_744_; lean_object* v___x_745_; lean_object* v___x_746_; lean_object* v___x_747_; 
v_typeName_740_ = lean_ctor_get(v_x_642_, 0);
lean_inc(v_typeName_740_);
v_idx_741_ = lean_ctor_get(v_x_642_, 1);
lean_inc(v_idx_741_);
v_struct_742_ = lean_ctor_get(v_x_642_, 2);
lean_inc_ref(v_struct_742_);
lean_dec_ref_known(v_x_642_, 3);
v___x_743_ = lean_obj_once(&lp_Qq_toExprExpr___closed__55, &lp_Qq_toExprExpr___closed__55_once, _init_lp_Qq_toExprExpr___closed__55);
v___x_744_ = l___private_Lean_ToExpr_0__Lean_Name_toExprAux(v_typeName_740_);
v___x_745_ = l_Lean_mkNatLit(v_idx_741_);
v___x_746_ = lp_Qq_toExprExpr(v_struct_742_);
v___x_747_ = l_Lean_mkApp3(v___x_743_, v___x_744_, v___x_745_, v___x_746_);
return v___x_747_;
}
}
}
}
LEAN_EXPORT lean_object* lp_Qq_List_forIn_x27_loop___at___00toExprExpr_spec__1(lean_object* v_as_748_, lean_object* v_as_x27_749_, lean_object* v_b_750_, lean_object* v_a_751_){
_start:
{
lean_object* v___x_752_; 
v___x_752_ = lp_Qq_List_forIn_x27_loop___at___00toExprExpr_spec__1___redArg(v_as_x27_749_, v_b_750_);
return v___x_752_;
}
}
LEAN_EXPORT lean_object* lp_Qq_List_forIn_x27_loop___at___00toExprExpr_spec__1___boxed(lean_object* v_as_753_, lean_object* v_as_x27_754_, lean_object* v_b_755_, lean_object* v_a_756_){
_start:
{
lean_object* v_res_757_; 
v_res_757_ = lp_Qq_List_forIn_x27_loop___at___00toExprExpr_spec__1(v_as_753_, v_as_x27_754_, v_b_755_, v_a_756_);
lean_dec(v_as_x27_754_);
lean_dec(v_as_753_);
return v_res_757_;
}
}
static lean_object* _init_lp_Qq_instToExprExpr__qq___closed__2(void){
_start:
{
lean_object* v___x_762_; lean_object* v___x_763_; lean_object* v___x_764_; 
v___x_762_ = lean_box(0);
v___x_763_ = ((lean_object*)(lp_Qq_instToExprExpr__qq___closed__1));
v___x_764_ = l_Lean_Expr_const___override(v___x_763_, v___x_762_);
return v___x_764_;
}
}
static lean_object* _init_lp_Qq_instToExprExpr__qq___closed__3(void){
_start:
{
lean_object* v___x_765_; lean_object* v___x_766_; lean_object* v___x_767_; 
v___x_765_ = lean_obj_once(&lp_Qq_instToExprExpr__qq___closed__2, &lp_Qq_instToExprExpr__qq___closed__2_once, _init_lp_Qq_instToExprExpr__qq___closed__2);
v___x_766_ = ((lean_object*)(lp_Qq_instToExprExpr__qq___closed__0));
v___x_767_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_767_, 0, v___x_766_);
lean_ctor_set(v___x_767_, 1, v___x_765_);
return v___x_767_;
}
}
static lean_object* _init_lp_Qq_instToExprExpr__qq(void){
_start:
{
lean_object* v___x_768_; 
v___x_768_ = lean_obj_once(&lp_Qq_instToExprExpr__qq___closed__3, &lp_Qq_instToExprExpr__qq___closed__3_once, _init_lp_Qq_instToExprExpr__qq___closed__3);
return v___x_768_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_Lean_ToExpr(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_Qq_Qq_ForLean_ToExpr(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_ToExpr(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_Qq_instToExprMVarId__qq = _init_lp_Qq_instToExprMVarId__qq();
lean_mark_persistent(lp_Qq_instToExprMVarId__qq);
lp_Qq_instToExprLevelMVarId__qq = _init_lp_Qq_instToExprLevelMVarId__qq();
lean_mark_persistent(lp_Qq_instToExprLevelMVarId__qq);
lp_Qq_instToExprLevel__qq = _init_lp_Qq_instToExprLevel__qq();
lean_mark_persistent(lp_Qq_instToExprLevel__qq);
lp_Qq_instToExprBinderInfo__qq = _init_lp_Qq_instToExprBinderInfo__qq();
lean_mark_persistent(lp_Qq_instToExprBinderInfo__qq);
lp_Qq_instToExprMData__qq = _init_lp_Qq_instToExprMData__qq();
lean_mark_persistent(lp_Qq_instToExprMData__qq);
lp_Qq_instToExprExpr__qq = _init_lp_Qq_instToExprExpr__qq();
lean_mark_persistent(lp_Qq_instToExprExpr__qq);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_Qq_Qq_ForLean_ToExpr(uint8_t builtin) {
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
lean_object* initialize_Lean_ToExpr(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_Qq_Qq_ForLean_ToExpr(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_ToExpr(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Qq_Qq_ForLean_ToExpr(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_Qq_Qq_ForLean_ToExpr(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_Qq_Qq_ForLean_ToExpr(builtin);
}
#ifdef __cplusplus
}
#endif
