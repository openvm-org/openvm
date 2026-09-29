// Lean compiler output
// Module: Mathlib.Algebra.Group.Equiv.Opposite
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Group.Equiv.Basic public import Mathlib.Algebra.Group.Opposite
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
lean_object* lp_mathlib_MulOpposite_opEquiv(lean_object*);
lean_object* lp_mathlib_AddOpposite_unop___boxed(lean_object*, lean_object*);
lean_object* l_Function_comp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_trans___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_AddOpposite_opEquiv(lean_object*);
lean_object* lp_mathlib_Equiv_symm___redArg(lean_object*);
lean_object* lp_mathlib_AddOpposite_op___boxed(lean_object*, lean_object*);
lean_object* lp_mathlib_MulOpposite_op___boxed(lean_object*, lean_object*);
lean_object* lp_mathlib_MulOpposite_unop___boxed(lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_MulOpposite_opMulEquiv___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_MulOpposite_opMulEquiv___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_opMulEquiv(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_opMulEquiv___boxed(lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_AddOpposite_opAddEquiv___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_AddOpposite_opAddEquiv___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_opAddEquiv(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_opAddEquiv___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_opAddEquiv(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_opAddEquiv___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_opMulEquiv(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_opMulEquiv___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_inv_x27___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_inv_x27___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_inv_x27(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_inv_x27___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_neg_x27___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_neg_x27___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_neg_x27(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_neg_x27___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulHom_toOpposite___redArg___lam__0(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_MulHom_toOpposite___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_MulOpposite_op___boxed, .m_arity = 2, .m_num_fixed = 1, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib_MulHom_toOpposite___redArg___closed__0 = (const lean_object*)&lp_mathlib_MulHom_toOpposite___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_MulHom_toOpposite___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulHom_toOpposite(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulHom_toOpposite___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_AddHom_toOpposite___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_AddOpposite_op___boxed, .m_arity = 2, .m_num_fixed = 1, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib_AddHom_toOpposite___redArg___closed__0 = (const lean_object*)&lp_mathlib_AddHom_toOpposite___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_AddHom_toOpposite___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddHom_toOpposite(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddHom_toOpposite___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_MulHom_fromOpposite___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_MulOpposite_unop___boxed, .m_arity = 2, .m_num_fixed = 1, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib_MulHom_fromOpposite___redArg___closed__0 = (const lean_object*)&lp_mathlib_MulHom_fromOpposite___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_MulHom_fromOpposite___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulHom_fromOpposite(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulHom_fromOpposite___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_AddHom_fromOpposite___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_AddOpposite_unop___boxed, .m_arity = 2, .m_num_fixed = 1, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib_AddHom_fromOpposite___redArg___closed__0 = (const lean_object*)&lp_mathlib_AddHom_fromOpposite___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_AddHom_fromOpposite___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddHom_fromOpposite(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddHom_fromOpposite___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_toOpposite___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_toOpposite(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_toOpposite___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_toOpposite___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_toOpposite(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_toOpposite___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_fromOpposite___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_fromOpposite(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_fromOpposite___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_fromOpposite___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_fromOpposite(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_fromOpposite___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_MulHom_op___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_MulHom_toOpposite___redArg___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_MulHom_op___closed__0 = (const lean_object*)&lp_mathlib_MulHom_op___closed__0_value;
static const lean_ctor_object lp_mathlib_MulHom_op___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_MulHom_op___closed__0_value),((lean_object*)&lp_mathlib_MulHom_op___closed__0_value)}};
static const lean_object* lp_mathlib_MulHom_op___closed__1 = (const lean_object*)&lp_mathlib_MulHom_op___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_MulHom_op(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulHom_op___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddHom_op(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddHom_op___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulHom_unop___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulHom_unop___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulHom_unop(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulHom_unop___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddHom_unop___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddHom_unop___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddHom_unop(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddHom_unop___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddHom_mulOp(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddHom_mulOp___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddHom_mulUnop___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddHom_mulUnop___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddHom_mulUnop(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddHom_mulUnop___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_op(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_op___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_op(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_op___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_unop___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_unop___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_unop(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_unop___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_unop___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_unop___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_unop(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_unop___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_MulEquiv_opOp___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_MulEquiv_opOp___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_opOp(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_opOp___boxed(lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_AddEquiv_opOp___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_AddEquiv_opOp___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_opOp(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_opOp___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_mulOp(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_mulOp___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_mulUnop___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_mulUnop___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_mulUnop(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_mulUnop___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_AddEquiv_mulOp___lam__0___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_AddEquiv_mulOp___lam__0___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_mulOp___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_mulOp___lam__1(lean_object*);
static const lean_closure_object lp_mathlib_AddEquiv_mulOp___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_AddEquiv_mulOp___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_AddEquiv_mulOp___closed__0 = (const lean_object*)&lp_mathlib_AddEquiv_mulOp___closed__0_value;
static const lean_closure_object lp_mathlib_AddEquiv_mulOp___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_AddEquiv_mulOp___lam__1, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_AddEquiv_mulOp___closed__1 = (const lean_object*)&lp_mathlib_AddEquiv_mulOp___closed__1_value;
static const lean_ctor_object lp_mathlib_AddEquiv_mulOp___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_AddEquiv_mulOp___closed__1_value),((lean_object*)&lp_mathlib_AddEquiv_mulOp___closed__0_value)}};
static const lean_object* lp_mathlib_AddEquiv_mulOp___closed__2 = (const lean_object*)&lp_mathlib_AddEquiv_mulOp___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_mulOp(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_mulOp___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_mulUnop___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_mulUnop___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_mulUnop(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_mulUnop___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_op___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_op___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_op___lam__2(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_op___lam__5(lean_object*);
static const lean_closure_object lp_mathlib_MulEquiv_op___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_MulEquiv_op___lam__2, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_MulEquiv_op___closed__0 = (const lean_object*)&lp_mathlib_MulEquiv_op___closed__0_value;
static const lean_closure_object lp_mathlib_MulEquiv_op___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_MulEquiv_op___lam__5, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_MulEquiv_op___closed__1 = (const lean_object*)&lp_mathlib_MulEquiv_op___closed__1_value;
static const lean_ctor_object lp_mathlib_MulEquiv_op___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_MulEquiv_op___closed__1_value),((lean_object*)&lp_mathlib_MulEquiv_op___closed__0_value)}};
static const lean_object* lp_mathlib_MulEquiv_op___closed__2 = (const lean_object*)&lp_mathlib_MulEquiv_op___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_op(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_op___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_op___lam__2(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_op___lam__3(lean_object*);
static const lean_closure_object lp_mathlib_AddEquiv_op___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_AddEquiv_op___lam__2, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_AddEquiv_op___closed__0 = (const lean_object*)&lp_mathlib_AddEquiv_op___closed__0_value;
static const lean_closure_object lp_mathlib_AddEquiv_op___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_AddEquiv_op___lam__3, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_AddEquiv_op___closed__1 = (const lean_object*)&lp_mathlib_AddEquiv_op___closed__1_value;
static const lean_ctor_object lp_mathlib_AddEquiv_op___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_AddEquiv_op___closed__1_value),((lean_object*)&lp_mathlib_AddEquiv_op___closed__0_value)}};
static const lean_object* lp_mathlib_AddEquiv_op___closed__2 = (const lean_object*)&lp_mathlib_AddEquiv_op___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_op(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_op___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_unop___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_unop___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_unop(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_unop___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_unop___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_unop___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_unop(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_unop___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_mathlib_MulOpposite_opMulEquiv___closed__0(void){
_start:
{
lean_object* v___x_1_; 
v___x_1_ = lp_mathlib_MulOpposite_opEquiv(lean_box(0));
return v___x_1_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_opMulEquiv(lean_object* v_M_2_, lean_object* v_inst_3_){
_start:
{
lean_object* v___x_4_; 
v___x_4_ = lean_obj_once(&lp_mathlib_MulOpposite_opMulEquiv___closed__0, &lp_mathlib_MulOpposite_opMulEquiv___closed__0_once, _init_lp_mathlib_MulOpposite_opMulEquiv___closed__0);
return v___x_4_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_opMulEquiv___boxed(lean_object* v_M_5_, lean_object* v_inst_6_){
_start:
{
lean_object* v_res_7_; 
v_res_7_ = lp_mathlib_MulOpposite_opMulEquiv(v_M_5_, v_inst_6_);
lean_dec_ref(v_inst_6_);
return v_res_7_;
}
}
static lean_object* _init_lp_mathlib_AddOpposite_opAddEquiv___closed__0(void){
_start:
{
lean_object* v___x_8_; 
v___x_8_ = lp_mathlib_AddOpposite_opEquiv(lean_box(0));
return v___x_8_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_opAddEquiv(lean_object* v_M_9_, lean_object* v_inst_10_){
_start:
{
lean_object* v___x_11_; 
v___x_11_ = lean_obj_once(&lp_mathlib_AddOpposite_opAddEquiv___closed__0, &lp_mathlib_AddOpposite_opAddEquiv___closed__0_once, _init_lp_mathlib_AddOpposite_opAddEquiv___closed__0);
return v___x_11_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_opAddEquiv___boxed(lean_object* v_M_12_, lean_object* v_inst_13_){
_start:
{
lean_object* v_res_14_; 
v_res_14_ = lp_mathlib_AddOpposite_opAddEquiv(v_M_12_, v_inst_13_);
lean_dec_ref(v_inst_13_);
return v_res_14_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_opAddEquiv(lean_object* v_00_u03b1_15_, lean_object* v_inst_16_){
_start:
{
lean_object* v___x_17_; 
v___x_17_ = lean_obj_once(&lp_mathlib_MulOpposite_opMulEquiv___closed__0, &lp_mathlib_MulOpposite_opMulEquiv___closed__0_once, _init_lp_mathlib_MulOpposite_opMulEquiv___closed__0);
return v___x_17_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_opAddEquiv___boxed(lean_object* v_00_u03b1_18_, lean_object* v_inst_19_){
_start:
{
lean_object* v_res_20_; 
v_res_20_ = lp_mathlib_MulOpposite_opAddEquiv(v_00_u03b1_18_, v_inst_19_);
lean_dec(v_inst_19_);
return v_res_20_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_opMulEquiv(lean_object* v_00_u03b1_21_, lean_object* v_inst_22_){
_start:
{
lean_object* v___x_23_; 
v___x_23_ = lean_obj_once(&lp_mathlib_AddOpposite_opAddEquiv___closed__0, &lp_mathlib_AddOpposite_opAddEquiv___closed__0_once, _init_lp_mathlib_AddOpposite_opAddEquiv___closed__0);
return v___x_23_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_opMulEquiv___boxed(lean_object* v_00_u03b1_24_, lean_object* v_inst_25_){
_start:
{
lean_object* v_res_26_; 
v_res_26_ = lp_mathlib_AddOpposite_opMulEquiv(v_00_u03b1_24_, v_inst_25_);
lean_dec(v_inst_25_);
return v_res_26_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_inv_x27___redArg(lean_object* v_inst_27_){
_start:
{
lean_object* v_toInv_28_; lean_object* v___x_29_; lean_object* v___x_30_; lean_object* v___x_31_; 
v_toInv_28_ = lean_ctor_get(v_inst_27_, 1);
lean_inc_n(v_toInv_28_, 2);
v___x_29_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_29_, 0, v_toInv_28_);
lean_ctor_set(v___x_29_, 1, v_toInv_28_);
v___x_30_ = lean_obj_once(&lp_mathlib_MulOpposite_opMulEquiv___closed__0, &lp_mathlib_MulOpposite_opMulEquiv___closed__0_once, _init_lp_mathlib_MulOpposite_opMulEquiv___closed__0);
v___x_31_ = lp_mathlib_Equiv_trans___redArg(v___x_29_, v___x_30_);
return v___x_31_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_inv_x27___redArg___boxed(lean_object* v_inst_32_){
_start:
{
lean_object* v_res_33_; 
v_res_33_ = lp_mathlib_MulEquiv_inv_x27___redArg(v_inst_32_);
lean_dec_ref(v_inst_32_);
return v_res_33_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_inv_x27(lean_object* v_G_34_, lean_object* v_inst_35_){
_start:
{
lean_object* v___x_36_; 
v___x_36_ = lp_mathlib_MulEquiv_inv_x27___redArg(v_inst_35_);
return v___x_36_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_inv_x27___boxed(lean_object* v_G_37_, lean_object* v_inst_38_){
_start:
{
lean_object* v_res_39_; 
v_res_39_ = lp_mathlib_MulEquiv_inv_x27(v_G_37_, v_inst_38_);
lean_dec_ref(v_inst_38_);
return v_res_39_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_neg_x27___redArg(lean_object* v_inst_40_){
_start:
{
lean_object* v_toNeg_41_; lean_object* v___x_42_; lean_object* v___x_43_; lean_object* v___x_44_; 
v_toNeg_41_ = lean_ctor_get(v_inst_40_, 1);
lean_inc_n(v_toNeg_41_, 2);
v___x_42_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_42_, 0, v_toNeg_41_);
lean_ctor_set(v___x_42_, 1, v_toNeg_41_);
v___x_43_ = lean_obj_once(&lp_mathlib_AddOpposite_opAddEquiv___closed__0, &lp_mathlib_AddOpposite_opAddEquiv___closed__0_once, _init_lp_mathlib_AddOpposite_opAddEquiv___closed__0);
v___x_44_ = lp_mathlib_Equiv_trans___redArg(v___x_42_, v___x_43_);
return v___x_44_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_neg_x27___redArg___boxed(lean_object* v_inst_45_){
_start:
{
lean_object* v_res_46_; 
v_res_46_ = lp_mathlib_AddEquiv_neg_x27___redArg(v_inst_45_);
lean_dec_ref(v_inst_45_);
return v_res_46_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_neg_x27(lean_object* v_G_47_, lean_object* v_inst_48_){
_start:
{
lean_object* v___x_49_; 
v___x_49_ = lp_mathlib_AddEquiv_neg_x27___redArg(v_inst_48_);
return v___x_49_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_neg_x27___boxed(lean_object* v_G_50_, lean_object* v_inst_51_){
_start:
{
lean_object* v_res_52_; 
v_res_52_ = lp_mathlib_AddEquiv_neg_x27(v_G_50_, v_inst_51_);
lean_dec_ref(v_inst_51_);
return v_res_52_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulHom_toOpposite___redArg___lam__0(lean_object* v_f_53_, lean_object* v___y_54_){
_start:
{
lean_object* v___x_55_; 
v___x_55_ = lean_apply_1(v_f_53_, v___y_54_);
return v___x_55_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulHom_toOpposite___redArg(lean_object* v_f_57_){
_start:
{
lean_object* v___f_58_; lean_object* v___x_59_; lean_object* v___x_60_; 
v___f_58_ = lean_alloc_closure((void*)(lp_mathlib_MulHom_toOpposite___redArg___lam__0), 2, 1);
lean_closure_set(v___f_58_, 0, v_f_57_);
v___x_59_ = ((lean_object*)(lp_mathlib_MulHom_toOpposite___redArg___closed__0));
v___x_60_ = lean_alloc_closure((void*)(l_Function_comp), 6, 5);
lean_closure_set(v___x_60_, 0, lean_box(0));
lean_closure_set(v___x_60_, 1, lean_box(0));
lean_closure_set(v___x_60_, 2, lean_box(0));
lean_closure_set(v___x_60_, 3, v___x_59_);
lean_closure_set(v___x_60_, 4, v___f_58_);
return v___x_60_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulHom_toOpposite(lean_object* v_M_61_, lean_object* v_N_62_, lean_object* v_inst_63_, lean_object* v_inst_64_, lean_object* v_f_65_, lean_object* v_hf_66_){
_start:
{
lean_object* v___x_67_; 
v___x_67_ = lp_mathlib_MulHom_toOpposite___redArg(v_f_65_);
return v___x_67_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulHom_toOpposite___boxed(lean_object* v_M_68_, lean_object* v_N_69_, lean_object* v_inst_70_, lean_object* v_inst_71_, lean_object* v_f_72_, lean_object* v_hf_73_){
_start:
{
lean_object* v_res_74_; 
v_res_74_ = lp_mathlib_MulHom_toOpposite(v_M_68_, v_N_69_, v_inst_70_, v_inst_71_, v_f_72_, v_hf_73_);
lean_dec(v_inst_71_);
lean_dec(v_inst_70_);
return v_res_74_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddHom_toOpposite___redArg(lean_object* v_f_76_){
_start:
{
lean_object* v___f_77_; lean_object* v___x_78_; lean_object* v___x_79_; 
v___f_77_ = lean_alloc_closure((void*)(lp_mathlib_MulHom_toOpposite___redArg___lam__0), 2, 1);
lean_closure_set(v___f_77_, 0, v_f_76_);
v___x_78_ = ((lean_object*)(lp_mathlib_AddHom_toOpposite___redArg___closed__0));
v___x_79_ = lean_alloc_closure((void*)(l_Function_comp), 6, 5);
lean_closure_set(v___x_79_, 0, lean_box(0));
lean_closure_set(v___x_79_, 1, lean_box(0));
lean_closure_set(v___x_79_, 2, lean_box(0));
lean_closure_set(v___x_79_, 3, v___x_78_);
lean_closure_set(v___x_79_, 4, v___f_77_);
return v___x_79_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddHom_toOpposite(lean_object* v_M_80_, lean_object* v_N_81_, lean_object* v_inst_82_, lean_object* v_inst_83_, lean_object* v_f_84_, lean_object* v_hf_85_){
_start:
{
lean_object* v___x_86_; 
v___x_86_ = lp_mathlib_AddHom_toOpposite___redArg(v_f_84_);
return v___x_86_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddHom_toOpposite___boxed(lean_object* v_M_87_, lean_object* v_N_88_, lean_object* v_inst_89_, lean_object* v_inst_90_, lean_object* v_f_91_, lean_object* v_hf_92_){
_start:
{
lean_object* v_res_93_; 
v_res_93_ = lp_mathlib_AddHom_toOpposite(v_M_87_, v_N_88_, v_inst_89_, v_inst_90_, v_f_91_, v_hf_92_);
lean_dec(v_inst_90_);
lean_dec(v_inst_89_);
return v_res_93_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulHom_fromOpposite___redArg(lean_object* v_f_95_){
_start:
{
lean_object* v___f_96_; lean_object* v___x_97_; lean_object* v___x_98_; 
v___f_96_ = lean_alloc_closure((void*)(lp_mathlib_MulHom_toOpposite___redArg___lam__0), 2, 1);
lean_closure_set(v___f_96_, 0, v_f_95_);
v___x_97_ = ((lean_object*)(lp_mathlib_MulHom_fromOpposite___redArg___closed__0));
v___x_98_ = lean_alloc_closure((void*)(l_Function_comp), 6, 5);
lean_closure_set(v___x_98_, 0, lean_box(0));
lean_closure_set(v___x_98_, 1, lean_box(0));
lean_closure_set(v___x_98_, 2, lean_box(0));
lean_closure_set(v___x_98_, 3, v___f_96_);
lean_closure_set(v___x_98_, 4, v___x_97_);
return v___x_98_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulHom_fromOpposite(lean_object* v_M_99_, lean_object* v_N_100_, lean_object* v_inst_101_, lean_object* v_inst_102_, lean_object* v_f_103_, lean_object* v_hf_104_){
_start:
{
lean_object* v___x_105_; 
v___x_105_ = lp_mathlib_MulHom_fromOpposite___redArg(v_f_103_);
return v___x_105_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulHom_fromOpposite___boxed(lean_object* v_M_106_, lean_object* v_N_107_, lean_object* v_inst_108_, lean_object* v_inst_109_, lean_object* v_f_110_, lean_object* v_hf_111_){
_start:
{
lean_object* v_res_112_; 
v_res_112_ = lp_mathlib_MulHom_fromOpposite(v_M_106_, v_N_107_, v_inst_108_, v_inst_109_, v_f_110_, v_hf_111_);
lean_dec(v_inst_109_);
lean_dec(v_inst_108_);
return v_res_112_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddHom_fromOpposite___redArg(lean_object* v_f_114_){
_start:
{
lean_object* v___f_115_; lean_object* v___x_116_; lean_object* v___x_117_; 
v___f_115_ = lean_alloc_closure((void*)(lp_mathlib_MulHom_toOpposite___redArg___lam__0), 2, 1);
lean_closure_set(v___f_115_, 0, v_f_114_);
v___x_116_ = ((lean_object*)(lp_mathlib_AddHom_fromOpposite___redArg___closed__0));
v___x_117_ = lean_alloc_closure((void*)(l_Function_comp), 6, 5);
lean_closure_set(v___x_117_, 0, lean_box(0));
lean_closure_set(v___x_117_, 1, lean_box(0));
lean_closure_set(v___x_117_, 2, lean_box(0));
lean_closure_set(v___x_117_, 3, v___f_115_);
lean_closure_set(v___x_117_, 4, v___x_116_);
return v___x_117_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddHom_fromOpposite(lean_object* v_M_118_, lean_object* v_N_119_, lean_object* v_inst_120_, lean_object* v_inst_121_, lean_object* v_f_122_, lean_object* v_hf_123_){
_start:
{
lean_object* v___x_124_; 
v___x_124_ = lp_mathlib_AddHom_fromOpposite___redArg(v_f_122_);
return v___x_124_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddHom_fromOpposite___boxed(lean_object* v_M_125_, lean_object* v_N_126_, lean_object* v_inst_127_, lean_object* v_inst_128_, lean_object* v_f_129_, lean_object* v_hf_130_){
_start:
{
lean_object* v_res_131_; 
v_res_131_ = lp_mathlib_AddHom_fromOpposite(v_M_125_, v_N_126_, v_inst_127_, v_inst_128_, v_f_129_, v_hf_130_);
lean_dec(v_inst_128_);
lean_dec(v_inst_127_);
return v_res_131_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_toOpposite___redArg(lean_object* v_f_132_){
_start:
{
lean_object* v___f_133_; lean_object* v___x_134_; lean_object* v___x_135_; 
v___f_133_ = lean_alloc_closure((void*)(lp_mathlib_MulHom_toOpposite___redArg___lam__0), 2, 1);
lean_closure_set(v___f_133_, 0, v_f_132_);
v___x_134_ = ((lean_object*)(lp_mathlib_MulHom_toOpposite___redArg___closed__0));
v___x_135_ = lean_alloc_closure((void*)(l_Function_comp), 6, 5);
lean_closure_set(v___x_135_, 0, lean_box(0));
lean_closure_set(v___x_135_, 1, lean_box(0));
lean_closure_set(v___x_135_, 2, lean_box(0));
lean_closure_set(v___x_135_, 3, v___x_134_);
lean_closure_set(v___x_135_, 4, v___f_133_);
return v___x_135_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_toOpposite(lean_object* v_M_136_, lean_object* v_N_137_, lean_object* v_inst_138_, lean_object* v_inst_139_, lean_object* v_f_140_, lean_object* v_hf_141_){
_start:
{
lean_object* v___x_142_; 
v___x_142_ = lp_mathlib_MonoidHom_toOpposite___redArg(v_f_140_);
return v___x_142_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_toOpposite___boxed(lean_object* v_M_143_, lean_object* v_N_144_, lean_object* v_inst_145_, lean_object* v_inst_146_, lean_object* v_f_147_, lean_object* v_hf_148_){
_start:
{
lean_object* v_res_149_; 
v_res_149_ = lp_mathlib_MonoidHom_toOpposite(v_M_143_, v_N_144_, v_inst_145_, v_inst_146_, v_f_147_, v_hf_148_);
lean_dec_ref(v_inst_146_);
lean_dec_ref(v_inst_145_);
return v_res_149_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_toOpposite___redArg(lean_object* v_f_150_){
_start:
{
lean_object* v___f_151_; lean_object* v___x_152_; lean_object* v___x_153_; 
v___f_151_ = lean_alloc_closure((void*)(lp_mathlib_MulHom_toOpposite___redArg___lam__0), 2, 1);
lean_closure_set(v___f_151_, 0, v_f_150_);
v___x_152_ = ((lean_object*)(lp_mathlib_AddHom_toOpposite___redArg___closed__0));
v___x_153_ = lean_alloc_closure((void*)(l_Function_comp), 6, 5);
lean_closure_set(v___x_153_, 0, lean_box(0));
lean_closure_set(v___x_153_, 1, lean_box(0));
lean_closure_set(v___x_153_, 2, lean_box(0));
lean_closure_set(v___x_153_, 3, v___x_152_);
lean_closure_set(v___x_153_, 4, v___f_151_);
return v___x_153_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_toOpposite(lean_object* v_M_154_, lean_object* v_N_155_, lean_object* v_inst_156_, lean_object* v_inst_157_, lean_object* v_f_158_, lean_object* v_hf_159_){
_start:
{
lean_object* v___x_160_; 
v___x_160_ = lp_mathlib_AddMonoidHom_toOpposite___redArg(v_f_158_);
return v___x_160_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_toOpposite___boxed(lean_object* v_M_161_, lean_object* v_N_162_, lean_object* v_inst_163_, lean_object* v_inst_164_, lean_object* v_f_165_, lean_object* v_hf_166_){
_start:
{
lean_object* v_res_167_; 
v_res_167_ = lp_mathlib_AddMonoidHom_toOpposite(v_M_161_, v_N_162_, v_inst_163_, v_inst_164_, v_f_165_, v_hf_166_);
lean_dec_ref(v_inst_164_);
lean_dec_ref(v_inst_163_);
return v_res_167_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_fromOpposite___redArg(lean_object* v_f_168_){
_start:
{
lean_object* v___f_169_; lean_object* v___x_170_; lean_object* v___x_171_; 
v___f_169_ = lean_alloc_closure((void*)(lp_mathlib_MulHom_toOpposite___redArg___lam__0), 2, 1);
lean_closure_set(v___f_169_, 0, v_f_168_);
v___x_170_ = ((lean_object*)(lp_mathlib_MulHom_fromOpposite___redArg___closed__0));
v___x_171_ = lean_alloc_closure((void*)(l_Function_comp), 6, 5);
lean_closure_set(v___x_171_, 0, lean_box(0));
lean_closure_set(v___x_171_, 1, lean_box(0));
lean_closure_set(v___x_171_, 2, lean_box(0));
lean_closure_set(v___x_171_, 3, v___f_169_);
lean_closure_set(v___x_171_, 4, v___x_170_);
return v___x_171_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_fromOpposite(lean_object* v_M_172_, lean_object* v_N_173_, lean_object* v_inst_174_, lean_object* v_inst_175_, lean_object* v_f_176_, lean_object* v_hf_177_){
_start:
{
lean_object* v___x_178_; 
v___x_178_ = lp_mathlib_MonoidHom_fromOpposite___redArg(v_f_176_);
return v___x_178_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_fromOpposite___boxed(lean_object* v_M_179_, lean_object* v_N_180_, lean_object* v_inst_181_, lean_object* v_inst_182_, lean_object* v_f_183_, lean_object* v_hf_184_){
_start:
{
lean_object* v_res_185_; 
v_res_185_ = lp_mathlib_MonoidHom_fromOpposite(v_M_179_, v_N_180_, v_inst_181_, v_inst_182_, v_f_183_, v_hf_184_);
lean_dec_ref(v_inst_182_);
lean_dec_ref(v_inst_181_);
return v_res_185_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_fromOpposite___redArg(lean_object* v_f_186_){
_start:
{
lean_object* v___f_187_; lean_object* v___x_188_; lean_object* v___x_189_; 
v___f_187_ = lean_alloc_closure((void*)(lp_mathlib_MulHom_toOpposite___redArg___lam__0), 2, 1);
lean_closure_set(v___f_187_, 0, v_f_186_);
v___x_188_ = ((lean_object*)(lp_mathlib_AddHom_fromOpposite___redArg___closed__0));
v___x_189_ = lean_alloc_closure((void*)(l_Function_comp), 6, 5);
lean_closure_set(v___x_189_, 0, lean_box(0));
lean_closure_set(v___x_189_, 1, lean_box(0));
lean_closure_set(v___x_189_, 2, lean_box(0));
lean_closure_set(v___x_189_, 3, v___f_187_);
lean_closure_set(v___x_189_, 4, v___x_188_);
return v___x_189_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_fromOpposite(lean_object* v_M_190_, lean_object* v_N_191_, lean_object* v_inst_192_, lean_object* v_inst_193_, lean_object* v_f_194_, lean_object* v_hf_195_){
_start:
{
lean_object* v___x_196_; 
v___x_196_ = lp_mathlib_AddMonoidHom_fromOpposite___redArg(v_f_194_);
return v___x_196_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_fromOpposite___boxed(lean_object* v_M_197_, lean_object* v_N_198_, lean_object* v_inst_199_, lean_object* v_inst_200_, lean_object* v_f_201_, lean_object* v_hf_202_){
_start:
{
lean_object* v_res_203_; 
v_res_203_ = lp_mathlib_AddMonoidHom_fromOpposite(v_M_197_, v_N_198_, v_inst_199_, v_inst_200_, v_f_201_, v_hf_202_);
lean_dec_ref(v_inst_200_);
lean_dec_ref(v_inst_199_);
return v_res_203_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulHom_op(lean_object* v_M_207_, lean_object* v_N_208_, lean_object* v_inst_209_, lean_object* v_inst_210_){
_start:
{
lean_object* v___x_211_; 
v___x_211_ = ((lean_object*)(lp_mathlib_MulHom_op___closed__1));
return v___x_211_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulHom_op___boxed(lean_object* v_M_212_, lean_object* v_N_213_, lean_object* v_inst_214_, lean_object* v_inst_215_){
_start:
{
lean_object* v_res_216_; 
v_res_216_ = lp_mathlib_MulHom_op(v_M_212_, v_N_213_, v_inst_214_, v_inst_215_);
lean_dec(v_inst_215_);
lean_dec(v_inst_214_);
return v_res_216_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddHom_op(lean_object* v_M_217_, lean_object* v_N_218_, lean_object* v_inst_219_, lean_object* v_inst_220_){
_start:
{
lean_object* v___x_221_; 
v___x_221_ = ((lean_object*)(lp_mathlib_MulHom_op___closed__1));
return v___x_221_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddHom_op___boxed(lean_object* v_M_222_, lean_object* v_N_223_, lean_object* v_inst_224_, lean_object* v_inst_225_){
_start:
{
lean_object* v_res_226_; 
v_res_226_ = lp_mathlib_AddHom_op(v_M_222_, v_N_223_, v_inst_224_, v_inst_225_);
lean_dec(v_inst_225_);
lean_dec(v_inst_224_);
return v_res_226_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulHom_unop___redArg(lean_object* v_inst_227_, lean_object* v_inst_228_){
_start:
{
lean_object* v___x_229_; lean_object* v___x_230_; 
v___x_229_ = lp_mathlib_MulHom_op(lean_box(0), lean_box(0), v_inst_227_, v_inst_228_);
v___x_230_ = lp_mathlib_Equiv_symm___redArg(v___x_229_);
return v___x_230_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulHom_unop___redArg___boxed(lean_object* v_inst_231_, lean_object* v_inst_232_){
_start:
{
lean_object* v_res_233_; 
v_res_233_ = lp_mathlib_MulHom_unop___redArg(v_inst_231_, v_inst_232_);
lean_dec(v_inst_232_);
lean_dec(v_inst_231_);
return v_res_233_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulHom_unop(lean_object* v_M_234_, lean_object* v_N_235_, lean_object* v_inst_236_, lean_object* v_inst_237_){
_start:
{
lean_object* v___x_238_; 
v___x_238_ = lp_mathlib_MulHom_unop___redArg(v_inst_236_, v_inst_237_);
return v___x_238_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulHom_unop___boxed(lean_object* v_M_239_, lean_object* v_N_240_, lean_object* v_inst_241_, lean_object* v_inst_242_){
_start:
{
lean_object* v_res_243_; 
v_res_243_ = lp_mathlib_MulHom_unop(v_M_239_, v_N_240_, v_inst_241_, v_inst_242_);
lean_dec(v_inst_242_);
lean_dec(v_inst_241_);
return v_res_243_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddHom_unop___redArg(lean_object* v_inst_244_, lean_object* v_inst_245_){
_start:
{
lean_object* v___x_246_; lean_object* v___x_247_; 
v___x_246_ = lp_mathlib_AddHom_op(lean_box(0), lean_box(0), v_inst_244_, v_inst_245_);
v___x_247_ = lp_mathlib_Equiv_symm___redArg(v___x_246_);
return v___x_247_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddHom_unop___redArg___boxed(lean_object* v_inst_248_, lean_object* v_inst_249_){
_start:
{
lean_object* v_res_250_; 
v_res_250_ = lp_mathlib_AddHom_unop___redArg(v_inst_248_, v_inst_249_);
lean_dec(v_inst_249_);
lean_dec(v_inst_248_);
return v_res_250_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddHom_unop(lean_object* v_M_251_, lean_object* v_N_252_, lean_object* v_inst_253_, lean_object* v_inst_254_){
_start:
{
lean_object* v___x_255_; 
v___x_255_ = lp_mathlib_AddHom_unop___redArg(v_inst_253_, v_inst_254_);
return v___x_255_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddHom_unop___boxed(lean_object* v_M_256_, lean_object* v_N_257_, lean_object* v_inst_258_, lean_object* v_inst_259_){
_start:
{
lean_object* v_res_260_; 
v_res_260_ = lp_mathlib_AddHom_unop(v_M_256_, v_N_257_, v_inst_258_, v_inst_259_);
lean_dec(v_inst_259_);
lean_dec(v_inst_258_);
return v_res_260_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddHom_mulOp(lean_object* v_M_261_, lean_object* v_N_262_, lean_object* v_inst_263_, lean_object* v_inst_264_){
_start:
{
lean_object* v___x_265_; 
v___x_265_ = ((lean_object*)(lp_mathlib_MulHom_op___closed__1));
return v___x_265_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddHom_mulOp___boxed(lean_object* v_M_266_, lean_object* v_N_267_, lean_object* v_inst_268_, lean_object* v_inst_269_){
_start:
{
lean_object* v_res_270_; 
v_res_270_ = lp_mathlib_AddHom_mulOp(v_M_266_, v_N_267_, v_inst_268_, v_inst_269_);
lean_dec(v_inst_269_);
lean_dec(v_inst_268_);
return v_res_270_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddHom_mulUnop___redArg(lean_object* v_inst_271_, lean_object* v_inst_272_){
_start:
{
lean_object* v___x_273_; lean_object* v___x_274_; 
v___x_273_ = lp_mathlib_AddHom_mulOp(lean_box(0), lean_box(0), v_inst_271_, v_inst_272_);
v___x_274_ = lp_mathlib_Equiv_symm___redArg(v___x_273_);
return v___x_274_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddHom_mulUnop___redArg___boxed(lean_object* v_inst_275_, lean_object* v_inst_276_){
_start:
{
lean_object* v_res_277_; 
v_res_277_ = lp_mathlib_AddHom_mulUnop___redArg(v_inst_275_, v_inst_276_);
lean_dec(v_inst_276_);
lean_dec(v_inst_275_);
return v_res_277_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddHom_mulUnop(lean_object* v_00_u03b1_278_, lean_object* v_00_u03b2_279_, lean_object* v_inst_280_, lean_object* v_inst_281_){
_start:
{
lean_object* v___x_282_; 
v___x_282_ = lp_mathlib_AddHom_mulUnop___redArg(v_inst_280_, v_inst_281_);
return v___x_282_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddHom_mulUnop___boxed(lean_object* v_00_u03b1_283_, lean_object* v_00_u03b2_284_, lean_object* v_inst_285_, lean_object* v_inst_286_){
_start:
{
lean_object* v_res_287_; 
v_res_287_ = lp_mathlib_AddHom_mulUnop(v_00_u03b1_283_, v_00_u03b2_284_, v_inst_285_, v_inst_286_);
lean_dec(v_inst_286_);
lean_dec(v_inst_285_);
return v_res_287_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_op(lean_object* v_M_288_, lean_object* v_N_289_, lean_object* v_inst_290_, lean_object* v_inst_291_){
_start:
{
lean_object* v___x_292_; 
v___x_292_ = ((lean_object*)(lp_mathlib_MulHom_op___closed__1));
return v___x_292_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_op___boxed(lean_object* v_M_293_, lean_object* v_N_294_, lean_object* v_inst_295_, lean_object* v_inst_296_){
_start:
{
lean_object* v_res_297_; 
v_res_297_ = lp_mathlib_MonoidHom_op(v_M_293_, v_N_294_, v_inst_295_, v_inst_296_);
lean_dec_ref(v_inst_296_);
lean_dec_ref(v_inst_295_);
return v_res_297_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_op(lean_object* v_M_298_, lean_object* v_N_299_, lean_object* v_inst_300_, lean_object* v_inst_301_){
_start:
{
lean_object* v___x_302_; 
v___x_302_ = ((lean_object*)(lp_mathlib_MulHom_op___closed__1));
return v___x_302_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_op___boxed(lean_object* v_M_303_, lean_object* v_N_304_, lean_object* v_inst_305_, lean_object* v_inst_306_){
_start:
{
lean_object* v_res_307_; 
v_res_307_ = lp_mathlib_AddMonoidHom_op(v_M_303_, v_N_304_, v_inst_305_, v_inst_306_);
lean_dec_ref(v_inst_306_);
lean_dec_ref(v_inst_305_);
return v_res_307_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_unop___redArg(lean_object* v_inst_308_, lean_object* v_inst_309_){
_start:
{
lean_object* v___x_310_; lean_object* v___x_311_; 
v___x_310_ = lp_mathlib_MonoidHom_op(lean_box(0), lean_box(0), v_inst_308_, v_inst_309_);
v___x_311_ = lp_mathlib_Equiv_symm___redArg(v___x_310_);
return v___x_311_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_unop___redArg___boxed(lean_object* v_inst_312_, lean_object* v_inst_313_){
_start:
{
lean_object* v_res_314_; 
v_res_314_ = lp_mathlib_MonoidHom_unop___redArg(v_inst_312_, v_inst_313_);
lean_dec_ref(v_inst_313_);
lean_dec_ref(v_inst_312_);
return v_res_314_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_unop(lean_object* v_M_315_, lean_object* v_N_316_, lean_object* v_inst_317_, lean_object* v_inst_318_){
_start:
{
lean_object* v___x_319_; 
v___x_319_ = lp_mathlib_MonoidHom_unop___redArg(v_inst_317_, v_inst_318_);
return v___x_319_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_unop___boxed(lean_object* v_M_320_, lean_object* v_N_321_, lean_object* v_inst_322_, lean_object* v_inst_323_){
_start:
{
lean_object* v_res_324_; 
v_res_324_ = lp_mathlib_MonoidHom_unop(v_M_320_, v_N_321_, v_inst_322_, v_inst_323_);
lean_dec_ref(v_inst_323_);
lean_dec_ref(v_inst_322_);
return v_res_324_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_unop___redArg(lean_object* v_inst_325_, lean_object* v_inst_326_){
_start:
{
lean_object* v___x_327_; lean_object* v___x_328_; 
v___x_327_ = lp_mathlib_AddMonoidHom_op(lean_box(0), lean_box(0), v_inst_325_, v_inst_326_);
v___x_328_ = lp_mathlib_Equiv_symm___redArg(v___x_327_);
return v___x_328_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_unop___redArg___boxed(lean_object* v_inst_329_, lean_object* v_inst_330_){
_start:
{
lean_object* v_res_331_; 
v_res_331_ = lp_mathlib_AddMonoidHom_unop___redArg(v_inst_329_, v_inst_330_);
lean_dec_ref(v_inst_330_);
lean_dec_ref(v_inst_329_);
return v_res_331_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_unop(lean_object* v_M_332_, lean_object* v_N_333_, lean_object* v_inst_334_, lean_object* v_inst_335_){
_start:
{
lean_object* v___x_336_; 
v___x_336_ = lp_mathlib_AddMonoidHom_unop___redArg(v_inst_334_, v_inst_335_);
return v___x_336_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_unop___boxed(lean_object* v_M_337_, lean_object* v_N_338_, lean_object* v_inst_339_, lean_object* v_inst_340_){
_start:
{
lean_object* v_res_341_; 
v_res_341_ = lp_mathlib_AddMonoidHom_unop(v_M_337_, v_N_338_, v_inst_339_, v_inst_340_);
lean_dec_ref(v_inst_340_);
lean_dec_ref(v_inst_339_);
return v_res_341_;
}
}
static lean_object* _init_lp_mathlib_MulEquiv_opOp___closed__0(void){
_start:
{
lean_object* v___x_342_; lean_object* v___x_343_; 
v___x_342_ = lean_obj_once(&lp_mathlib_MulOpposite_opMulEquiv___closed__0, &lp_mathlib_MulOpposite_opMulEquiv___closed__0_once, _init_lp_mathlib_MulOpposite_opMulEquiv___closed__0);
v___x_343_ = lp_mathlib_Equiv_trans___redArg(v___x_342_, v___x_342_);
return v___x_343_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_opOp(lean_object* v_M_344_, lean_object* v_inst_345_){
_start:
{
lean_object* v___x_346_; 
v___x_346_ = lean_obj_once(&lp_mathlib_MulEquiv_opOp___closed__0, &lp_mathlib_MulEquiv_opOp___closed__0_once, _init_lp_mathlib_MulEquiv_opOp___closed__0);
return v___x_346_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_opOp___boxed(lean_object* v_M_347_, lean_object* v_inst_348_){
_start:
{
lean_object* v_res_349_; 
v_res_349_ = lp_mathlib_MulEquiv_opOp(v_M_347_, v_inst_348_);
lean_dec(v_inst_348_);
return v_res_349_;
}
}
static lean_object* _init_lp_mathlib_AddEquiv_opOp___closed__0(void){
_start:
{
lean_object* v___x_350_; lean_object* v___x_351_; 
v___x_350_ = lean_obj_once(&lp_mathlib_AddOpposite_opAddEquiv___closed__0, &lp_mathlib_AddOpposite_opAddEquiv___closed__0_once, _init_lp_mathlib_AddOpposite_opAddEquiv___closed__0);
v___x_351_ = lp_mathlib_Equiv_trans___redArg(v___x_350_, v___x_350_);
return v___x_351_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_opOp(lean_object* v_M_352_, lean_object* v_inst_353_){
_start:
{
lean_object* v___x_354_; 
v___x_354_ = lean_obj_once(&lp_mathlib_AddEquiv_opOp___closed__0, &lp_mathlib_AddEquiv_opOp___closed__0_once, _init_lp_mathlib_AddEquiv_opOp___closed__0);
return v___x_354_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_opOp___boxed(lean_object* v_M_355_, lean_object* v_inst_356_){
_start:
{
lean_object* v_res_357_; 
v_res_357_ = lp_mathlib_AddEquiv_opOp(v_M_355_, v_inst_356_);
lean_dec(v_inst_356_);
return v_res_357_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_mulOp(lean_object* v_M_358_, lean_object* v_N_359_, lean_object* v_inst_360_, lean_object* v_inst_361_){
_start:
{
lean_object* v___x_362_; 
v___x_362_ = ((lean_object*)(lp_mathlib_MulHom_op___closed__1));
return v___x_362_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_mulOp___boxed(lean_object* v_M_363_, lean_object* v_N_364_, lean_object* v_inst_365_, lean_object* v_inst_366_){
_start:
{
lean_object* v_res_367_; 
v_res_367_ = lp_mathlib_AddMonoidHom_mulOp(v_M_363_, v_N_364_, v_inst_365_, v_inst_366_);
lean_dec_ref(v_inst_366_);
lean_dec_ref(v_inst_365_);
return v_res_367_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_mulUnop___redArg(lean_object* v_inst_368_, lean_object* v_inst_369_){
_start:
{
lean_object* v___x_370_; lean_object* v___x_371_; 
v___x_370_ = lp_mathlib_AddMonoidHom_mulOp(lean_box(0), lean_box(0), v_inst_368_, v_inst_369_);
v___x_371_ = lp_mathlib_Equiv_symm___redArg(v___x_370_);
return v___x_371_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_mulUnop___redArg___boxed(lean_object* v_inst_372_, lean_object* v_inst_373_){
_start:
{
lean_object* v_res_374_; 
v_res_374_ = lp_mathlib_AddMonoidHom_mulUnop___redArg(v_inst_372_, v_inst_373_);
lean_dec_ref(v_inst_373_);
lean_dec_ref(v_inst_372_);
return v_res_374_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_mulUnop(lean_object* v_00_u03b1_375_, lean_object* v_00_u03b2_376_, lean_object* v_inst_377_, lean_object* v_inst_378_){
_start:
{
lean_object* v___x_379_; 
v___x_379_ = lp_mathlib_AddMonoidHom_mulUnop___redArg(v_inst_377_, v_inst_378_);
return v___x_379_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_mulUnop___boxed(lean_object* v_00_u03b1_380_, lean_object* v_00_u03b2_381_, lean_object* v_inst_382_, lean_object* v_inst_383_){
_start:
{
lean_object* v_res_384_; 
v_res_384_ = lp_mathlib_AddMonoidHom_mulUnop(v_00_u03b1_380_, v_00_u03b2_381_, v_inst_382_, v_inst_383_);
lean_dec_ref(v_inst_383_);
lean_dec_ref(v_inst_382_);
return v_res_384_;
}
}
static lean_object* _init_lp_mathlib_AddEquiv_mulOp___lam__0___closed__0(void){
_start:
{
lean_object* v___x_385_; lean_object* v___x_386_; 
v___x_385_ = lean_obj_once(&lp_mathlib_MulOpposite_opMulEquiv___closed__0, &lp_mathlib_MulOpposite_opMulEquiv___closed__0_once, _init_lp_mathlib_MulOpposite_opMulEquiv___closed__0);
v___x_386_ = lp_mathlib_Equiv_symm___redArg(v___x_385_);
return v___x_386_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_mulOp___lam__0(lean_object* v_f_387_){
_start:
{
lean_object* v___x_388_; lean_object* v___x_389_; lean_object* v___x_390_; lean_object* v___x_391_; 
v___x_388_ = lean_obj_once(&lp_mathlib_MulOpposite_opMulEquiv___closed__0, &lp_mathlib_MulOpposite_opMulEquiv___closed__0_once, _init_lp_mathlib_MulOpposite_opMulEquiv___closed__0);
v___x_389_ = lean_obj_once(&lp_mathlib_AddEquiv_mulOp___lam__0___closed__0, &lp_mathlib_AddEquiv_mulOp___lam__0___closed__0_once, _init_lp_mathlib_AddEquiv_mulOp___lam__0___closed__0);
v___x_390_ = lp_mathlib_Equiv_trans___redArg(v_f_387_, v___x_389_);
v___x_391_ = lp_mathlib_Equiv_trans___redArg(v___x_388_, v___x_390_);
return v___x_391_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_mulOp___lam__1(lean_object* v_f_392_){
_start:
{
lean_object* v___x_393_; lean_object* v___x_394_; lean_object* v___x_395_; lean_object* v___x_396_; 
v___x_393_ = lean_obj_once(&lp_mathlib_MulOpposite_opMulEquiv___closed__0, &lp_mathlib_MulOpposite_opMulEquiv___closed__0_once, _init_lp_mathlib_MulOpposite_opMulEquiv___closed__0);
v___x_394_ = lean_obj_once(&lp_mathlib_AddEquiv_mulOp___lam__0___closed__0, &lp_mathlib_AddEquiv_mulOp___lam__0___closed__0_once, _init_lp_mathlib_AddEquiv_mulOp___lam__0___closed__0);
v___x_395_ = lp_mathlib_Equiv_trans___redArg(v_f_392_, v___x_393_);
v___x_396_ = lp_mathlib_Equiv_trans___redArg(v___x_394_, v___x_395_);
return v___x_396_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_mulOp(lean_object* v_00_u03b1_402_, lean_object* v_00_u03b2_403_, lean_object* v_inst_404_, lean_object* v_inst_405_){
_start:
{
lean_object* v___x_406_; 
v___x_406_ = ((lean_object*)(lp_mathlib_AddEquiv_mulOp___closed__2));
return v___x_406_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_mulOp___boxed(lean_object* v_00_u03b1_407_, lean_object* v_00_u03b2_408_, lean_object* v_inst_409_, lean_object* v_inst_410_){
_start:
{
lean_object* v_res_411_; 
v_res_411_ = lp_mathlib_AddEquiv_mulOp(v_00_u03b1_407_, v_00_u03b2_408_, v_inst_409_, v_inst_410_);
lean_dec(v_inst_410_);
lean_dec(v_inst_409_);
return v_res_411_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_mulUnop___redArg(lean_object* v_inst_412_, lean_object* v_inst_413_){
_start:
{
lean_object* v___x_414_; lean_object* v___x_415_; 
v___x_414_ = lp_mathlib_AddEquiv_mulOp(lean_box(0), lean_box(0), v_inst_412_, v_inst_413_);
v___x_415_ = lp_mathlib_Equiv_symm___redArg(v___x_414_);
return v___x_415_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_mulUnop___redArg___boxed(lean_object* v_inst_416_, lean_object* v_inst_417_){
_start:
{
lean_object* v_res_418_; 
v_res_418_ = lp_mathlib_AddEquiv_mulUnop___redArg(v_inst_416_, v_inst_417_);
lean_dec(v_inst_417_);
lean_dec(v_inst_416_);
return v_res_418_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_mulUnop(lean_object* v_00_u03b1_419_, lean_object* v_00_u03b2_420_, lean_object* v_inst_421_, lean_object* v_inst_422_){
_start:
{
lean_object* v___x_423_; 
v___x_423_ = lp_mathlib_AddEquiv_mulUnop___redArg(v_inst_421_, v_inst_422_);
return v___x_423_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_mulUnop___boxed(lean_object* v_00_u03b1_424_, lean_object* v_00_u03b2_425_, lean_object* v_inst_426_, lean_object* v_inst_427_){
_start:
{
lean_object* v_res_428_; 
v_res_428_ = lp_mathlib_AddEquiv_mulUnop(v_00_u03b1_424_, v_00_u03b2_425_, v_inst_426_, v_inst_427_);
lean_dec(v_inst_427_);
lean_dec(v_inst_426_);
return v_res_428_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_op___lam__0(lean_object* v_f_429_, lean_object* v___y_430_){
_start:
{
lean_object* v_toFun_431_; lean_object* v___x_432_; 
v_toFun_431_ = lean_ctor_get(v_f_429_, 0);
lean_inc(v_toFun_431_);
lean_dec_ref(v_f_429_);
v___x_432_ = lean_apply_1(v_toFun_431_, v___y_430_);
return v___x_432_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_op___lam__1(lean_object* v___x_433_, lean_object* v___y_434_){
_start:
{
lean_object* v_toFun_435_; lean_object* v___x_436_; 
v_toFun_435_ = lean_ctor_get(v___x_433_, 0);
lean_inc(v_toFun_435_);
lean_dec_ref(v___x_433_);
v___x_436_ = lean_apply_1(v_toFun_435_, v___y_434_);
return v___x_436_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_op___lam__2(lean_object* v_f_437_){
_start:
{
lean_object* v___f_438_; lean_object* v___x_439_; lean_object* v___x_440_; lean_object* v___x_441_; lean_object* v___x_442_; lean_object* v___x_443_; lean_object* v___f_444_; lean_object* v___x_445_; lean_object* v___x_446_; lean_object* v___x_447_; 
lean_inc_ref(v_f_437_);
v___f_438_ = lean_alloc_closure((void*)(lp_mathlib_MulEquiv_op___lam__0), 2, 1);
lean_closure_set(v___f_438_, 0, v_f_437_);
v___x_439_ = ((lean_object*)(lp_mathlib_MulHom_fromOpposite___redArg___closed__0));
v___x_440_ = ((lean_object*)(lp_mathlib_MulHom_toOpposite___redArg___closed__0));
v___x_441_ = lean_alloc_closure((void*)(l_Function_comp), 6, 5);
lean_closure_set(v___x_441_, 0, lean_box(0));
lean_closure_set(v___x_441_, 1, lean_box(0));
lean_closure_set(v___x_441_, 2, lean_box(0));
lean_closure_set(v___x_441_, 3, v___f_438_);
lean_closure_set(v___x_441_, 4, v___x_440_);
v___x_442_ = lean_alloc_closure((void*)(l_Function_comp), 6, 5);
lean_closure_set(v___x_442_, 0, lean_box(0));
lean_closure_set(v___x_442_, 1, lean_box(0));
lean_closure_set(v___x_442_, 2, lean_box(0));
lean_closure_set(v___x_442_, 3, v___x_439_);
lean_closure_set(v___x_442_, 4, v___x_441_);
v___x_443_ = lp_mathlib_Equiv_symm___redArg(v_f_437_);
v___f_444_ = lean_alloc_closure((void*)(lp_mathlib_MulEquiv_op___lam__1), 2, 1);
lean_closure_set(v___f_444_, 0, v___x_443_);
v___x_445_ = lean_alloc_closure((void*)(l_Function_comp), 6, 5);
lean_closure_set(v___x_445_, 0, lean_box(0));
lean_closure_set(v___x_445_, 1, lean_box(0));
lean_closure_set(v___x_445_, 2, lean_box(0));
lean_closure_set(v___x_445_, 3, v___f_444_);
lean_closure_set(v___x_445_, 4, v___x_440_);
v___x_446_ = lean_alloc_closure((void*)(l_Function_comp), 6, 5);
lean_closure_set(v___x_446_, 0, lean_box(0));
lean_closure_set(v___x_446_, 1, lean_box(0));
lean_closure_set(v___x_446_, 2, lean_box(0));
lean_closure_set(v___x_446_, 3, v___x_439_);
lean_closure_set(v___x_446_, 4, v___x_445_);
v___x_447_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_447_, 0, v___x_442_);
lean_ctor_set(v___x_447_, 1, v___x_446_);
return v___x_447_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_op___lam__5(lean_object* v_f_448_){
_start:
{
lean_object* v___f_449_; lean_object* v___x_450_; lean_object* v___x_451_; lean_object* v___x_452_; lean_object* v___x_453_; lean_object* v___x_454_; lean_object* v___f_455_; lean_object* v___x_456_; lean_object* v___x_457_; lean_object* v___x_458_; 
lean_inc_ref(v_f_448_);
v___f_449_ = lean_alloc_closure((void*)(lp_mathlib_MulEquiv_op___lam__0), 2, 1);
lean_closure_set(v___f_449_, 0, v_f_448_);
v___x_450_ = ((lean_object*)(lp_mathlib_MulHom_toOpposite___redArg___closed__0));
v___x_451_ = ((lean_object*)(lp_mathlib_MulHom_fromOpposite___redArg___closed__0));
v___x_452_ = lean_alloc_closure((void*)(l_Function_comp), 6, 5);
lean_closure_set(v___x_452_, 0, lean_box(0));
lean_closure_set(v___x_452_, 1, lean_box(0));
lean_closure_set(v___x_452_, 2, lean_box(0));
lean_closure_set(v___x_452_, 3, v___f_449_);
lean_closure_set(v___x_452_, 4, v___x_451_);
v___x_453_ = lean_alloc_closure((void*)(l_Function_comp), 6, 5);
lean_closure_set(v___x_453_, 0, lean_box(0));
lean_closure_set(v___x_453_, 1, lean_box(0));
lean_closure_set(v___x_453_, 2, lean_box(0));
lean_closure_set(v___x_453_, 3, v___x_450_);
lean_closure_set(v___x_453_, 4, v___x_452_);
v___x_454_ = lp_mathlib_Equiv_symm___redArg(v_f_448_);
v___f_455_ = lean_alloc_closure((void*)(lp_mathlib_MulEquiv_op___lam__1), 2, 1);
lean_closure_set(v___f_455_, 0, v___x_454_);
v___x_456_ = lean_alloc_closure((void*)(l_Function_comp), 6, 5);
lean_closure_set(v___x_456_, 0, lean_box(0));
lean_closure_set(v___x_456_, 1, lean_box(0));
lean_closure_set(v___x_456_, 2, lean_box(0));
lean_closure_set(v___x_456_, 3, v___f_455_);
lean_closure_set(v___x_456_, 4, v___x_451_);
v___x_457_ = lean_alloc_closure((void*)(l_Function_comp), 6, 5);
lean_closure_set(v___x_457_, 0, lean_box(0));
lean_closure_set(v___x_457_, 1, lean_box(0));
lean_closure_set(v___x_457_, 2, lean_box(0));
lean_closure_set(v___x_457_, 3, v___x_450_);
lean_closure_set(v___x_457_, 4, v___x_456_);
v___x_458_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_458_, 0, v___x_453_);
lean_ctor_set(v___x_458_, 1, v___x_457_);
return v___x_458_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_op(lean_object* v_00_u03b1_464_, lean_object* v_00_u03b2_465_, lean_object* v_inst_466_, lean_object* v_inst_467_){
_start:
{
lean_object* v___x_468_; 
v___x_468_ = ((lean_object*)(lp_mathlib_MulEquiv_op___closed__2));
return v___x_468_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_op___boxed(lean_object* v_00_u03b1_469_, lean_object* v_00_u03b2_470_, lean_object* v_inst_471_, lean_object* v_inst_472_){
_start:
{
lean_object* v_res_473_; 
v_res_473_ = lp_mathlib_MulEquiv_op(v_00_u03b1_469_, v_00_u03b2_470_, v_inst_471_, v_inst_472_);
lean_dec(v_inst_472_);
lean_dec(v_inst_471_);
return v_res_473_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_op___lam__2(lean_object* v_f_474_){
_start:
{
lean_object* v___f_475_; lean_object* v___x_476_; lean_object* v___x_477_; lean_object* v___x_478_; lean_object* v___x_479_; lean_object* v___x_480_; lean_object* v___f_481_; lean_object* v___x_482_; lean_object* v___x_483_; lean_object* v___x_484_; 
lean_inc_ref(v_f_474_);
v___f_475_ = lean_alloc_closure((void*)(lp_mathlib_MulEquiv_op___lam__0), 2, 1);
lean_closure_set(v___f_475_, 0, v_f_474_);
v___x_476_ = ((lean_object*)(lp_mathlib_AddHom_fromOpposite___redArg___closed__0));
v___x_477_ = ((lean_object*)(lp_mathlib_AddHom_toOpposite___redArg___closed__0));
v___x_478_ = lean_alloc_closure((void*)(l_Function_comp), 6, 5);
lean_closure_set(v___x_478_, 0, lean_box(0));
lean_closure_set(v___x_478_, 1, lean_box(0));
lean_closure_set(v___x_478_, 2, lean_box(0));
lean_closure_set(v___x_478_, 3, v___f_475_);
lean_closure_set(v___x_478_, 4, v___x_477_);
v___x_479_ = lean_alloc_closure((void*)(l_Function_comp), 6, 5);
lean_closure_set(v___x_479_, 0, lean_box(0));
lean_closure_set(v___x_479_, 1, lean_box(0));
lean_closure_set(v___x_479_, 2, lean_box(0));
lean_closure_set(v___x_479_, 3, v___x_476_);
lean_closure_set(v___x_479_, 4, v___x_478_);
v___x_480_ = lp_mathlib_Equiv_symm___redArg(v_f_474_);
v___f_481_ = lean_alloc_closure((void*)(lp_mathlib_MulEquiv_op___lam__1), 2, 1);
lean_closure_set(v___f_481_, 0, v___x_480_);
v___x_482_ = lean_alloc_closure((void*)(l_Function_comp), 6, 5);
lean_closure_set(v___x_482_, 0, lean_box(0));
lean_closure_set(v___x_482_, 1, lean_box(0));
lean_closure_set(v___x_482_, 2, lean_box(0));
lean_closure_set(v___x_482_, 3, v___f_481_);
lean_closure_set(v___x_482_, 4, v___x_477_);
v___x_483_ = lean_alloc_closure((void*)(l_Function_comp), 6, 5);
lean_closure_set(v___x_483_, 0, lean_box(0));
lean_closure_set(v___x_483_, 1, lean_box(0));
lean_closure_set(v___x_483_, 2, lean_box(0));
lean_closure_set(v___x_483_, 3, v___x_476_);
lean_closure_set(v___x_483_, 4, v___x_482_);
v___x_484_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_484_, 0, v___x_479_);
lean_ctor_set(v___x_484_, 1, v___x_483_);
return v___x_484_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_op___lam__3(lean_object* v_f_485_){
_start:
{
lean_object* v___f_486_; lean_object* v___x_487_; lean_object* v___x_488_; lean_object* v___x_489_; lean_object* v___x_490_; lean_object* v___x_491_; lean_object* v___f_492_; lean_object* v___x_493_; lean_object* v___x_494_; lean_object* v___x_495_; 
lean_inc_ref(v_f_485_);
v___f_486_ = lean_alloc_closure((void*)(lp_mathlib_MulEquiv_op___lam__0), 2, 1);
lean_closure_set(v___f_486_, 0, v_f_485_);
v___x_487_ = ((lean_object*)(lp_mathlib_AddHom_toOpposite___redArg___closed__0));
v___x_488_ = ((lean_object*)(lp_mathlib_AddHom_fromOpposite___redArg___closed__0));
v___x_489_ = lean_alloc_closure((void*)(l_Function_comp), 6, 5);
lean_closure_set(v___x_489_, 0, lean_box(0));
lean_closure_set(v___x_489_, 1, lean_box(0));
lean_closure_set(v___x_489_, 2, lean_box(0));
lean_closure_set(v___x_489_, 3, v___f_486_);
lean_closure_set(v___x_489_, 4, v___x_488_);
v___x_490_ = lean_alloc_closure((void*)(l_Function_comp), 6, 5);
lean_closure_set(v___x_490_, 0, lean_box(0));
lean_closure_set(v___x_490_, 1, lean_box(0));
lean_closure_set(v___x_490_, 2, lean_box(0));
lean_closure_set(v___x_490_, 3, v___x_487_);
lean_closure_set(v___x_490_, 4, v___x_489_);
v___x_491_ = lp_mathlib_Equiv_symm___redArg(v_f_485_);
v___f_492_ = lean_alloc_closure((void*)(lp_mathlib_MulEquiv_op___lam__1), 2, 1);
lean_closure_set(v___f_492_, 0, v___x_491_);
v___x_493_ = lean_alloc_closure((void*)(l_Function_comp), 6, 5);
lean_closure_set(v___x_493_, 0, lean_box(0));
lean_closure_set(v___x_493_, 1, lean_box(0));
lean_closure_set(v___x_493_, 2, lean_box(0));
lean_closure_set(v___x_493_, 3, v___f_492_);
lean_closure_set(v___x_493_, 4, v___x_488_);
v___x_494_ = lean_alloc_closure((void*)(l_Function_comp), 6, 5);
lean_closure_set(v___x_494_, 0, lean_box(0));
lean_closure_set(v___x_494_, 1, lean_box(0));
lean_closure_set(v___x_494_, 2, lean_box(0));
lean_closure_set(v___x_494_, 3, v___x_487_);
lean_closure_set(v___x_494_, 4, v___x_493_);
v___x_495_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_495_, 0, v___x_490_);
lean_ctor_set(v___x_495_, 1, v___x_494_);
return v___x_495_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_op(lean_object* v_00_u03b1_501_, lean_object* v_00_u03b2_502_, lean_object* v_inst_503_, lean_object* v_inst_504_){
_start:
{
lean_object* v___x_505_; 
v___x_505_ = ((lean_object*)(lp_mathlib_AddEquiv_op___closed__2));
return v___x_505_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_op___boxed(lean_object* v_00_u03b1_506_, lean_object* v_00_u03b2_507_, lean_object* v_inst_508_, lean_object* v_inst_509_){
_start:
{
lean_object* v_res_510_; 
v_res_510_ = lp_mathlib_AddEquiv_op(v_00_u03b1_506_, v_00_u03b2_507_, v_inst_508_, v_inst_509_);
lean_dec(v_inst_509_);
lean_dec(v_inst_508_);
return v_res_510_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_unop___redArg(lean_object* v_inst_511_, lean_object* v_inst_512_){
_start:
{
lean_object* v___x_513_; lean_object* v___x_514_; 
v___x_513_ = lp_mathlib_MulEquiv_op(lean_box(0), lean_box(0), v_inst_511_, v_inst_512_);
v___x_514_ = lp_mathlib_Equiv_symm___redArg(v___x_513_);
return v___x_514_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_unop___redArg___boxed(lean_object* v_inst_515_, lean_object* v_inst_516_){
_start:
{
lean_object* v_res_517_; 
v_res_517_ = lp_mathlib_MulEquiv_unop___redArg(v_inst_515_, v_inst_516_);
lean_dec(v_inst_516_);
lean_dec(v_inst_515_);
return v_res_517_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_unop(lean_object* v_00_u03b1_518_, lean_object* v_00_u03b2_519_, lean_object* v_inst_520_, lean_object* v_inst_521_){
_start:
{
lean_object* v___x_522_; 
v___x_522_ = lp_mathlib_MulEquiv_unop___redArg(v_inst_520_, v_inst_521_);
return v___x_522_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_unop___boxed(lean_object* v_00_u03b1_523_, lean_object* v_00_u03b2_524_, lean_object* v_inst_525_, lean_object* v_inst_526_){
_start:
{
lean_object* v_res_527_; 
v_res_527_ = lp_mathlib_MulEquiv_unop(v_00_u03b1_523_, v_00_u03b2_524_, v_inst_525_, v_inst_526_);
lean_dec(v_inst_526_);
lean_dec(v_inst_525_);
return v_res_527_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_unop___redArg(lean_object* v_inst_528_, lean_object* v_inst_529_){
_start:
{
lean_object* v___x_530_; lean_object* v___x_531_; 
v___x_530_ = lp_mathlib_AddEquiv_op(lean_box(0), lean_box(0), v_inst_528_, v_inst_529_);
v___x_531_ = lp_mathlib_Equiv_symm___redArg(v___x_530_);
return v___x_531_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_unop___redArg___boxed(lean_object* v_inst_532_, lean_object* v_inst_533_){
_start:
{
lean_object* v_res_534_; 
v_res_534_ = lp_mathlib_AddEquiv_unop___redArg(v_inst_532_, v_inst_533_);
lean_dec(v_inst_533_);
lean_dec(v_inst_532_);
return v_res_534_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_unop(lean_object* v_00_u03b1_535_, lean_object* v_00_u03b2_536_, lean_object* v_inst_537_, lean_object* v_inst_538_){
_start:
{
lean_object* v___x_539_; 
v___x_539_ = lp_mathlib_AddEquiv_unop___redArg(v_inst_537_, v_inst_538_);
return v___x_539_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_unop___boxed(lean_object* v_00_u03b1_540_, lean_object* v_00_u03b2_541_, lean_object* v_inst_542_, lean_object* v_inst_543_){
_start:
{
lean_object* v_res_544_; 
v_res_544_ = lp_mathlib_AddEquiv_unop(v_00_u03b1_540_, v_00_u03b2_541_, v_inst_542_, v_inst_543_);
lean_dec(v_inst_543_);
lean_dec(v_inst_542_);
return v_res_544_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Equiv_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Opposite(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Equiv_Opposite(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Equiv_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Opposite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_Group_Equiv_Opposite(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Equiv_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Opposite(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Group_Equiv_Opposite(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Equiv_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Opposite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Equiv_Opposite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_Group_Equiv_Opposite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_Group_Equiv_Opposite(builtin);
}
#ifdef __cplusplus
}
#endif
