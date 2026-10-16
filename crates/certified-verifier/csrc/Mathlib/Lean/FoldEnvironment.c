// Lean compiler output
// Module: Mathlib.Lean.FoldEnvironment
// Imports: public import Init public meta import Init public import Lean.Meta.Basic public import Mathlib.Init
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
lean_object* l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lean_st_ref_get(lean_object*);
lean_object* l_Lean_Meta_Config_toConfigWithKey(lean_object*);
lean_object* lean_st_mk_ref(lean_object*);
lean_object* l_Lean_Environment_header(lean_object*);
lean_object* lean_array_get_size(lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
lean_object* lean_array_fget_borrowed(lean_object*, lean_object*);
lean_object* lean_st_ref_take(lean_object*);
lean_object* l_Lean_Name_num___override(lean_object*, lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
lean_object* lean_io_get_num_heartbeats();
extern lean_object* l_Lean_firstFrontendMacroScope;
extern lean_object* l_Lean_NameSet_empty;
extern lean_object* l_Lean_inheritedTraceOptions;
extern lean_object* l_Lean_KVMap_instValueBool;
extern lean_object* l_Lean_KVMap_instValueNat;
extern lean_object* l_Lean_instInhabitedConstantInfo_default;
extern lean_object* l_Lean_instInhabitedModuleData_default;
lean_object* lean_array_get(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_EnvironmentHeader_moduleNames(lean_object*);
lean_object* lean_array_get_borrowed(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* l_Lean_MessageData_ofName(lean_object*);
lean_object* l_Lean_Exception_toMessageData(lean_object*);
uint8_t l_Lean_Exception_isInterrupt(lean_object*);
uint8_t l_Lean_Exception_isRuntime(lean_object*);
lean_object* l_WellFounded_opaqueFix_u2083___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_instMonadEIO(lean_object*);
lean_object* l_StateRefT_x27_instMonad___redArg(lean_object*);
lean_object* l_Lean_Core_instMonadCoreM___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Core_instMonadCoreM___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instFunctorOfMonad___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instFunctorOfMonad___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instApplicativeOfMonad___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instApplicativeOfMonad___redArg___lam__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instApplicativeOfMonad___redArg___lam__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_IO_CancelToken_isSet(lean_object*);
lean_object* l_instMonadExceptOfEIO(lean_object*);
lean_object* l_StateRefT_x27_instMonadExceptOf___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_StateRefT_x27_instMonadExceptOf___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instMonadExceptOf___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instMonadExceptOf___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_Core_instMonadRefCoreM;
extern lean_object* l_Lean_Core_instAddMessageContextCoreM;
lean_object* l_Lean_instAddErrorMessageContextOfAddMessageContextOfMonad___redArg(lean_object*, lean_object*);
lean_object* l_Lean_throwInterruptException___redArg(lean_object*);
extern lean_object* l_Lean_diagnostics;
lean_object* l_Lean_Option_get___redArg(lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_maxRecDepth;
lean_object* l_Lean_Kernel_enableDiag(lean_object*, uint8_t);
uint8_t l_Lean_Kernel_isDiagnosticsEnabled(lean_object*);
lean_object* l_EIO_toBaseIO___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_io_as_task(lean_object*, lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* l_Lean_Meta_instMonadMetaM___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_instMonadMetaM___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Environment_constants(lean_object*);
lean_object* l_Lean_PersistentHashMap_foldlMAux___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_visitConst___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 25, .m_capacity = 25, .m_length = 24, .m_data = "Processing failure with "};
static const lean_object* lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_visitConst___redArg___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_visitConst___redArg___closed__0_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_visitConst___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_visitConst___redArg___closed__1;
static const lean_string_object lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_visitConst___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = " in "};
static const lean_object* lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_visitConst___redArg___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_visitConst___redArg___closed__2_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_visitConst___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_visitConst___redArg___closed__3;
static const lean_string_object lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_visitConst___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = ":\n  "};
static const lean_object* lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_visitConst___redArg___closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_visitConst___redArg___closed__4_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_visitConst___redArg___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_visitConst___redArg___closed__5;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_visitConst___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_visitConst___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_visitConst(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_visitConst___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___lam__0___boxed(lean_object**);
static lean_once_cell_t lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___lam__1___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___lam__1___closed__0;
static lean_once_cell_t lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___lam__1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___lam__1___closed__1;
static const lean_closure_object lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___lam__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Core_instMonadCoreM___lam__0___boxed, .m_arity = 5, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___lam__1___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___lam__1___closed__2_value;
static const lean_closure_object lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___lam__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Core_instMonadCoreM___lam__1___boxed, .m_arity = 7, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___lam__1___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___lam__1___closed__3_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___lam__1___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___lam__1___closed__4;
static lean_once_cell_t lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___lam__1___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___lam__1___closed__5;
static lean_once_cell_t lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___lam__1___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___lam__1___closed__6;
static lean_once_cell_t lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___lam__1___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___lam__1___closed__7;
static lean_once_cell_t lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___lam__1___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___lam__1___closed__8;
static lean_once_cell_t lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___lam__1___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___lam__1___closed__9;
static lean_once_cell_t lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___lam__1___closed__10_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___lam__1___closed__10;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___lam__1___boxed(lean_object**);
static lean_once_cell_t lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___closed__0;
static lean_once_cell_t lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___closed__1;
static lean_once_cell_t lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___closed__2;
static lean_once_cell_t lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___closed__3;
static lean_once_cell_t lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___closed__4;
static lean_once_cell_t lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___closed__5;
static lean_once_cell_t lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___closed__6;
static lean_once_cell_t lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___closed__7;
static const lean_ctor_object lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(1) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___closed__8 = (const lean_object*)&lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___closed__8_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___closed__9;
static lean_once_cell_t lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___closed__10_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___closed__10;
static lean_once_cell_t lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___closed__11_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___closed__11;
static lean_once_cell_t lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___closed__12_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___closed__12;
static const lean_array_object lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___closed__13 = (const lean_object*)&lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___closed__13_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___closed__14_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___closed__14;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_foldImportedDecls___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_foldImportedDecls___redArg___lam__0___boxed(lean_object**);
static lean_once_cell_t lp_mathlib_Lean_Meta_foldImportedDecls___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Meta_foldImportedDecls___redArg___closed__0;
static lean_once_cell_t lp_mathlib_Lean_Meta_foldImportedDecls___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Meta_foldImportedDecls___redArg___closed__1;
static lean_once_cell_t lp_mathlib_Lean_Meta_foldImportedDecls___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Meta_foldImportedDecls___redArg___closed__2;
static lean_once_cell_t lp_mathlib_Lean_Meta_foldImportedDecls___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Meta_foldImportedDecls___redArg___closed__3;
static lean_once_cell_t lp_mathlib_Lean_Meta_foldImportedDecls___redArg___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Meta_foldImportedDecls___redArg___closed__4;
static const lean_array_object lp_mathlib_Lean_Meta_foldImportedDecls___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Lean_Meta_foldImportedDecls___redArg___closed__5 = (const lean_object*)&lp_mathlib_Lean_Meta_foldImportedDecls___redArg___closed__5_value;
static const lean_ctor_object lp_mathlib_Lean_Meta_foldImportedDecls___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Lean_Meta_foldImportedDecls___redArg___closed__6 = (const lean_object*)&lp_mathlib_Lean_Meta_foldImportedDecls___redArg___closed__6_value;
static const lean_ctor_object lp_mathlib_Lean_Meta_foldImportedDecls___redArg___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Meta_foldImportedDecls___redArg___closed__5_value),((lean_object*)&lp_mathlib_Lean_Meta_foldImportedDecls___redArg___closed__6_value)}};
static const lean_object* lp_mathlib_Lean_Meta_foldImportedDecls___redArg___closed__7 = (const lean_object*)&lp_mathlib_Lean_Meta_foldImportedDecls___redArg___closed__7_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_foldImportedDecls___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_foldImportedDecls___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_foldImportedDecls(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_foldImportedDecls___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Lean_Meta_foldCurrFileDecls___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Meta_instMonadMetaM___lam__0___boxed, .m_arity = 7, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Lean_Meta_foldCurrFileDecls___redArg___closed__0 = (const lean_object*)&lp_mathlib_Lean_Meta_foldCurrFileDecls___redArg___closed__0_value;
static const lean_closure_object lp_mathlib_Lean_Meta_foldCurrFileDecls___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Meta_instMonadMetaM___lam__1___boxed, .m_arity = 9, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Lean_Meta_foldCurrFileDecls___redArg___closed__1 = (const lean_object*)&lp_mathlib_Lean_Meta_foldCurrFileDecls___redArg___closed__1_value;
static lean_once_cell_t lp_mathlib_Lean_Meta_foldCurrFileDecls___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Meta_foldCurrFileDecls___redArg___closed__2;
static lean_once_cell_t lp_mathlib_Lean_Meta_foldCurrFileDecls___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Meta_foldCurrFileDecls___redArg___closed__3;
static lean_once_cell_t lp_mathlib_Lean_Meta_foldCurrFileDecls___redArg___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Meta_foldCurrFileDecls___redArg___closed__4;
static const lean_ctor_object lp_mathlib_Lean_Meta_foldCurrFileDecls___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(1) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Lean_Meta_foldCurrFileDecls___redArg___closed__5 = (const lean_object*)&lp_mathlib_Lean_Meta_foldCurrFileDecls___redArg___closed__5_value;
static lean_once_cell_t lp_mathlib_Lean_Meta_foldCurrFileDecls___redArg___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Meta_foldCurrFileDecls___redArg___closed__6;
static lean_once_cell_t lp_mathlib_Lean_Meta_foldCurrFileDecls___redArg___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Meta_foldCurrFileDecls___redArg___closed__7;
static lean_once_cell_t lp_mathlib_Lean_Meta_foldCurrFileDecls___redArg___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Meta_foldCurrFileDecls___redArg___closed__8;
static lean_once_cell_t lp_mathlib_Lean_Meta_foldCurrFileDecls___redArg___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Meta_foldCurrFileDecls___redArg___closed__9;
static lean_once_cell_t lp_mathlib_Lean_Meta_foldCurrFileDecls___redArg___closed__10_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Meta_foldCurrFileDecls___redArg___closed__10;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_foldCurrFileDecls___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_foldCurrFileDecls___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_foldCurrFileDecls(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_foldCurrFileDecls___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_visitConst___redArg___closed__1(void){
_start:
{
lean_object* v___x_2_; lean_object* v___x_3_; 
v___x_2_ = ((lean_object*)(lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_visitConst___redArg___closed__0));
v___x_3_ = l_Lean_stringToMessageData(v___x_2_);
return v___x_3_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_visitConst___redArg___closed__3(void){
_start:
{
lean_object* v___x_5_; lean_object* v___x_6_; 
v___x_5_ = ((lean_object*)(lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_visitConst___redArg___closed__2));
v___x_6_ = l_Lean_stringToMessageData(v___x_5_);
return v___x_6_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_visitConst___redArg___closed__5(void){
_start:
{
lean_object* v___x_8_; lean_object* v___x_9_; 
v___x_8_ = ((lean_object*)(lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_visitConst___redArg___closed__4));
v___x_9_ = l_Lean_stringToMessageData(v___x_8_);
return v___x_9_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_visitConst___redArg(lean_object* v_modName_10_, lean_object* v_errorRef_11_, lean_object* v_act_12_, lean_object* v_a_13_, lean_object* v_name_14_, lean_object* v_constInfo_15_, lean_object* v_a_16_, lean_object* v_a_17_, lean_object* v_a_18_, lean_object* v_a_19_){
_start:
{
lean_object* v___x_21_; 
lean_inc(v_a_19_);
lean_inc_ref(v_a_18_);
lean_inc(v_a_17_);
lean_inc_ref(v_a_16_);
lean_inc(v_name_14_);
lean_inc(v_a_13_);
v___x_21_ = lean_apply_8(v_act_12_, v_a_13_, v_name_14_, v_constInfo_15_, v_a_16_, v_a_17_, v_a_18_, v_a_19_, lean_box(0));
if (lean_obj_tag(v___x_21_) == 0)
{
lean_dec(v_name_14_);
lean_dec(v_a_13_);
lean_dec(v_modName_10_);
return v___x_21_;
}
else
{
lean_object* v_a_22_; uint8_t v___y_24_; uint8_t v___x_47_; 
v_a_22_ = lean_ctor_get(v___x_21_, 0);
lean_inc(v_a_22_);
v___x_47_ = l_Lean_Exception_isInterrupt(v_a_22_);
if (v___x_47_ == 0)
{
uint8_t v___x_48_; 
lean_inc(v_a_22_);
v___x_48_ = l_Lean_Exception_isRuntime(v_a_22_);
v___y_24_ = v___x_48_;
goto v___jp_23_;
}
else
{
v___y_24_ = v___x_47_;
goto v___jp_23_;
}
v___jp_23_:
{
if (v___y_24_ == 0)
{
lean_object* v___x_26_; uint8_t v_isShared_27_; uint8_t v_isSharedCheck_45_; 
v_isSharedCheck_45_ = !lean_is_exclusive(v___x_21_);
if (v_isSharedCheck_45_ == 0)
{
lean_object* v_unused_46_; 
v_unused_46_ = lean_ctor_get(v___x_21_, 0);
lean_dec(v_unused_46_);
v___x_26_ = v___x_21_;
v_isShared_27_ = v_isSharedCheck_45_;
goto v_resetjp_25_;
}
else
{
lean_dec(v___x_21_);
v___x_26_ = lean_box(0);
v_isShared_27_ = v_isSharedCheck_45_;
goto v_resetjp_25_;
}
v_resetjp_25_:
{
lean_object* v___x_28_; lean_object* v___x_29_; lean_object* v___x_30_; lean_object* v___x_31_; lean_object* v___x_32_; lean_object* v___x_33_; lean_object* v___x_34_; lean_object* v___x_35_; lean_object* v___x_36_; lean_object* v___x_37_; lean_object* v___x_38_; lean_object* v___x_39_; lean_object* v___x_40_; lean_object* v___x_41_; lean_object* v___x_43_; 
v___x_28_ = lean_st_ref_take(v_errorRef_11_);
v___x_29_ = lean_obj_once(&lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_visitConst___redArg___closed__1, &lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_visitConst___redArg___closed__1_once, _init_lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_visitConst___redArg___closed__1);
v___x_30_ = l_Lean_MessageData_ofName(v_name_14_);
v___x_31_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_31_, 0, v___x_29_);
lean_ctor_set(v___x_31_, 1, v___x_30_);
v___x_32_ = lean_obj_once(&lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_visitConst___redArg___closed__3, &lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_visitConst___redArg___closed__3_once, _init_lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_visitConst___redArg___closed__3);
v___x_33_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_33_, 0, v___x_31_);
lean_ctor_set(v___x_33_, 1, v___x_32_);
v___x_34_ = l_Lean_MessageData_ofName(v_modName_10_);
v___x_35_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_35_, 0, v___x_33_);
lean_ctor_set(v___x_35_, 1, v___x_34_);
v___x_36_ = lean_obj_once(&lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_visitConst___redArg___closed__5, &lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_visitConst___redArg___closed__5_once, _init_lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_visitConst___redArg___closed__5);
v___x_37_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_37_, 0, v___x_35_);
lean_ctor_set(v___x_37_, 1, v___x_36_);
v___x_38_ = l_Lean_Exception_toMessageData(v_a_22_);
v___x_39_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_39_, 0, v___x_37_);
lean_ctor_set(v___x_39_, 1, v___x_38_);
v___x_40_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_40_, 0, v___x_39_);
lean_ctor_set(v___x_40_, 1, v___x_28_);
v___x_41_ = lean_st_ref_set(v_errorRef_11_, v___x_40_);
if (v_isShared_27_ == 0)
{
lean_ctor_set_tag(v___x_26_, 0);
lean_ctor_set(v___x_26_, 0, v_a_13_);
v___x_43_ = v___x_26_;
goto v_reusejp_42_;
}
else
{
lean_object* v_reuseFailAlloc_44_; 
v_reuseFailAlloc_44_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_44_, 0, v_a_13_);
v___x_43_ = v_reuseFailAlloc_44_;
goto v_reusejp_42_;
}
v_reusejp_42_:
{
return v___x_43_;
}
}
}
else
{
lean_dec(v_a_22_);
lean_dec(v_name_14_);
lean_dec(v_a_13_);
lean_dec(v_modName_10_);
return v___x_21_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_visitConst___redArg___boxed(lean_object* v_modName_49_, lean_object* v_errorRef_50_, lean_object* v_act_51_, lean_object* v_a_52_, lean_object* v_name_53_, lean_object* v_constInfo_54_, lean_object* v_a_55_, lean_object* v_a_56_, lean_object* v_a_57_, lean_object* v_a_58_, lean_object* v_a_59_){
_start:
{
lean_object* v_res_60_; 
v_res_60_ = lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_visitConst___redArg(v_modName_49_, v_errorRef_50_, v_act_51_, v_a_52_, v_name_53_, v_constInfo_54_, v_a_55_, v_a_56_, v_a_57_, v_a_58_);
lean_dec(v_a_58_);
lean_dec_ref(v_a_57_);
lean_dec(v_a_56_);
lean_dec_ref(v_a_55_);
lean_dec(v_errorRef_50_);
return v_res_60_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_visitConst(lean_object* v_00_u03b1_61_, lean_object* v_modName_62_, lean_object* v_errorRef_63_, lean_object* v_act_64_, lean_object* v_a_65_, lean_object* v_name_66_, lean_object* v_constInfo_67_, lean_object* v_a_68_, lean_object* v_a_69_, lean_object* v_a_70_, lean_object* v_a_71_){
_start:
{
lean_object* v___x_73_; 
lean_inc(v_a_71_);
lean_inc_ref(v_a_70_);
lean_inc(v_a_69_);
lean_inc_ref(v_a_68_);
lean_inc(v_name_66_);
lean_inc(v_a_65_);
v___x_73_ = lean_apply_8(v_act_64_, v_a_65_, v_name_66_, v_constInfo_67_, v_a_68_, v_a_69_, v_a_70_, v_a_71_, lean_box(0));
if (lean_obj_tag(v___x_73_) == 0)
{
lean_dec(v_name_66_);
lean_dec(v_a_65_);
lean_dec(v_modName_62_);
return v___x_73_;
}
else
{
lean_object* v_a_74_; uint8_t v___y_76_; uint8_t v___x_99_; 
v_a_74_ = lean_ctor_get(v___x_73_, 0);
lean_inc(v_a_74_);
v___x_99_ = l_Lean_Exception_isInterrupt(v_a_74_);
if (v___x_99_ == 0)
{
uint8_t v___x_100_; 
lean_inc(v_a_74_);
v___x_100_ = l_Lean_Exception_isRuntime(v_a_74_);
v___y_76_ = v___x_100_;
goto v___jp_75_;
}
else
{
v___y_76_ = v___x_99_;
goto v___jp_75_;
}
v___jp_75_:
{
if (v___y_76_ == 0)
{
lean_object* v___x_78_; uint8_t v_isShared_79_; uint8_t v_isSharedCheck_97_; 
v_isSharedCheck_97_ = !lean_is_exclusive(v___x_73_);
if (v_isSharedCheck_97_ == 0)
{
lean_object* v_unused_98_; 
v_unused_98_ = lean_ctor_get(v___x_73_, 0);
lean_dec(v_unused_98_);
v___x_78_ = v___x_73_;
v_isShared_79_ = v_isSharedCheck_97_;
goto v_resetjp_77_;
}
else
{
lean_dec(v___x_73_);
v___x_78_ = lean_box(0);
v_isShared_79_ = v_isSharedCheck_97_;
goto v_resetjp_77_;
}
v_resetjp_77_:
{
lean_object* v___x_80_; lean_object* v___x_81_; lean_object* v___x_82_; lean_object* v___x_83_; lean_object* v___x_84_; lean_object* v___x_85_; lean_object* v___x_86_; lean_object* v___x_87_; lean_object* v___x_88_; lean_object* v___x_89_; lean_object* v___x_90_; lean_object* v___x_91_; lean_object* v___x_92_; lean_object* v___x_93_; lean_object* v___x_95_; 
v___x_80_ = lean_st_ref_take(v_errorRef_63_);
v___x_81_ = lean_obj_once(&lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_visitConst___redArg___closed__1, &lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_visitConst___redArg___closed__1_once, _init_lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_visitConst___redArg___closed__1);
v___x_82_ = l_Lean_MessageData_ofName(v_name_66_);
v___x_83_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_83_, 0, v___x_81_);
lean_ctor_set(v___x_83_, 1, v___x_82_);
v___x_84_ = lean_obj_once(&lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_visitConst___redArg___closed__3, &lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_visitConst___redArg___closed__3_once, _init_lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_visitConst___redArg___closed__3);
v___x_85_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_85_, 0, v___x_83_);
lean_ctor_set(v___x_85_, 1, v___x_84_);
v___x_86_ = l_Lean_MessageData_ofName(v_modName_62_);
v___x_87_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_87_, 0, v___x_85_);
lean_ctor_set(v___x_87_, 1, v___x_86_);
v___x_88_ = lean_obj_once(&lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_visitConst___redArg___closed__5, &lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_visitConst___redArg___closed__5_once, _init_lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_visitConst___redArg___closed__5);
v___x_89_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_89_, 0, v___x_87_);
lean_ctor_set(v___x_89_, 1, v___x_88_);
v___x_90_ = l_Lean_Exception_toMessageData(v_a_74_);
v___x_91_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_91_, 0, v___x_89_);
lean_ctor_set(v___x_91_, 1, v___x_90_);
v___x_92_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_92_, 0, v___x_91_);
lean_ctor_set(v___x_92_, 1, v___x_80_);
v___x_93_ = lean_st_ref_set(v_errorRef_63_, v___x_92_);
if (v_isShared_79_ == 0)
{
lean_ctor_set_tag(v___x_78_, 0);
lean_ctor_set(v___x_78_, 0, v_a_65_);
v___x_95_ = v___x_78_;
goto v_reusejp_94_;
}
else
{
lean_object* v_reuseFailAlloc_96_; 
v_reuseFailAlloc_96_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_96_, 0, v_a_65_);
v___x_95_ = v_reuseFailAlloc_96_;
goto v_reusejp_94_;
}
v_reusejp_94_:
{
return v___x_95_;
}
}
}
else
{
lean_dec(v_a_74_);
lean_dec(v_name_66_);
lean_dec(v_a_65_);
lean_dec(v_modName_62_);
return v___x_73_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_visitConst___boxed(lean_object* v_00_u03b1_101_, lean_object* v_modName_102_, lean_object* v_errorRef_103_, lean_object* v_act_104_, lean_object* v_a_105_, lean_object* v_name_106_, lean_object* v_constInfo_107_, lean_object* v_a_108_, lean_object* v_a_109_, lean_object* v_a_110_, lean_object* v_a_111_, lean_object* v_a_112_){
_start:
{
lean_object* v_res_113_; 
v_res_113_ = lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_visitConst(v_00_u03b1_101_, v_modName_102_, v_errorRef_103_, v_act_104_, v_a_105_, v_name_106_, v_constInfo_107_, v_a_108_, v_a_109_, v_a_110_, v_a_111_);
lean_dec(v_a_111_);
lean_dec_ref(v_a_110_);
lean_dec(v_a_109_);
lean_dec_ref(v_a_108_);
lean_dec(v_errorRef_103_);
return v_res_113_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___lam__0(lean_object* v___x_114_, lean_object* v___x_115_, lean_object* v_constNames_116_, lean_object* v___x_117_, lean_object* v_constants_118_, lean_object* v_act_119_, lean_object* v_errorRef_120_, lean_object* v___x_121_, lean_object* v_next_122_, lean_object* v_acc_123_, lean_object* v_h_124_, lean_object* v_G_125_, lean_object* v___y_126_, lean_object* v___y_127_, lean_object* v___y_128_, lean_object* v___y_129_){
_start:
{
lean_object* v_a_132_; uint8_t v___x_135_; 
v___x_135_ = lean_nat_dec_lt(v_next_122_, v___x_114_);
if (v___x_135_ == 0)
{
lean_object* v___x_136_; 
lean_dec_ref(v_G_125_);
lean_dec(v___x_121_);
lean_dec_ref(v_act_119_);
v___x_136_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_136_, 0, v_acc_123_);
return v___x_136_;
}
else
{
lean_object* v___x_137_; lean_object* v___x_138_; lean_object* v___x_139_; 
v___x_137_ = lean_array_fget_borrowed(v_constNames_116_, v_next_122_);
v___x_138_ = lean_array_get_borrowed(v___x_117_, v_constants_118_, v_next_122_);
lean_inc(v___y_129_);
lean_inc_ref(v___y_128_);
lean_inc(v___y_127_);
lean_inc_ref(v___y_126_);
lean_inc(v___x_138_);
lean_inc(v___x_137_);
lean_inc(v_acc_123_);
v___x_139_ = lean_apply_8(v_act_119_, v_acc_123_, v___x_137_, v___x_138_, v___y_126_, v___y_127_, v___y_128_, v___y_129_, lean_box(0));
if (lean_obj_tag(v___x_139_) == 0)
{
lean_object* v_a_140_; 
lean_dec(v_acc_123_);
lean_dec(v___x_121_);
v_a_140_ = lean_ctor_get(v___x_139_, 0);
lean_inc(v_a_140_);
lean_dec_ref_known(v___x_139_, 1);
v_a_132_ = v_a_140_;
goto v___jp_131_;
}
else
{
lean_object* v_a_141_; uint8_t v___y_143_; uint8_t v___x_158_; 
v_a_141_ = lean_ctor_get(v___x_139_, 0);
lean_inc(v_a_141_);
v___x_158_ = l_Lean_Exception_isInterrupt(v_a_141_);
if (v___x_158_ == 0)
{
uint8_t v___x_159_; 
lean_inc(v_a_141_);
v___x_159_ = l_Lean_Exception_isRuntime(v_a_141_);
v___y_143_ = v___x_159_;
goto v___jp_142_;
}
else
{
v___y_143_ = v___x_158_;
goto v___jp_142_;
}
v___jp_142_:
{
if (v___y_143_ == 0)
{
lean_object* v___x_144_; lean_object* v___x_145_; lean_object* v___x_146_; lean_object* v___x_147_; lean_object* v___x_148_; lean_object* v___x_149_; lean_object* v___x_150_; lean_object* v___x_151_; lean_object* v___x_152_; lean_object* v___x_153_; lean_object* v___x_154_; lean_object* v___x_155_; lean_object* v___x_156_; lean_object* v___x_157_; 
lean_dec_ref_known(v___x_139_, 1);
v___x_144_ = lean_st_ref_take(v_errorRef_120_);
v___x_145_ = lean_obj_once(&lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_visitConst___redArg___closed__1, &lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_visitConst___redArg___closed__1_once, _init_lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_visitConst___redArg___closed__1);
lean_inc(v___x_137_);
v___x_146_ = l_Lean_MessageData_ofName(v___x_137_);
v___x_147_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_147_, 0, v___x_145_);
lean_ctor_set(v___x_147_, 1, v___x_146_);
v___x_148_ = lean_obj_once(&lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_visitConst___redArg___closed__3, &lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_visitConst___redArg___closed__3_once, _init_lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_visitConst___redArg___closed__3);
v___x_149_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_149_, 0, v___x_147_);
lean_ctor_set(v___x_149_, 1, v___x_148_);
v___x_150_ = l_Lean_MessageData_ofName(v___x_121_);
v___x_151_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_151_, 0, v___x_149_);
lean_ctor_set(v___x_151_, 1, v___x_150_);
v___x_152_ = lean_obj_once(&lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_visitConst___redArg___closed__5, &lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_visitConst___redArg___closed__5_once, _init_lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_visitConst___redArg___closed__5);
v___x_153_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_153_, 0, v___x_151_);
lean_ctor_set(v___x_153_, 1, v___x_152_);
v___x_154_ = l_Lean_Exception_toMessageData(v_a_141_);
v___x_155_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_155_, 0, v___x_153_);
lean_ctor_set(v___x_155_, 1, v___x_154_);
v___x_156_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_156_, 0, v___x_155_);
lean_ctor_set(v___x_156_, 1, v___x_144_);
v___x_157_ = lean_st_ref_set(v_errorRef_120_, v___x_156_);
v_a_132_ = v_acc_123_;
goto v___jp_131_;
}
else
{
lean_dec(v_a_141_);
lean_dec_ref(v_G_125_);
lean_dec(v_acc_123_);
lean_dec(v___x_121_);
return v___x_139_;
}
}
}
}
v___jp_131_:
{
lean_object* v___x_133_; lean_object* v___x_134_; 
v___x_133_ = lean_nat_add(v_next_122_, v___x_115_);
lean_inc(v___y_129_);
lean_inc_ref(v___y_128_);
lean_inc(v___y_127_);
lean_inc_ref(v___y_126_);
v___x_134_ = lean_apply_9(v_G_125_, v___x_133_, v_a_132_, lean_box(0), lean_box(0), v___y_126_, v___y_127_, v___y_128_, v___y_129_, lean_box(0));
return v___x_134_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___lam__0___boxed(lean_object** _args){
lean_object* v___x_160_ = _args[0];
lean_object* v___x_161_ = _args[1];
lean_object* v_constNames_162_ = _args[2];
lean_object* v___x_163_ = _args[3];
lean_object* v_constants_164_ = _args[4];
lean_object* v_act_165_ = _args[5];
lean_object* v_errorRef_166_ = _args[6];
lean_object* v___x_167_ = _args[7];
lean_object* v_next_168_ = _args[8];
lean_object* v_acc_169_ = _args[9];
lean_object* v_h_170_ = _args[10];
lean_object* v_G_171_ = _args[11];
lean_object* v___y_172_ = _args[12];
lean_object* v___y_173_ = _args[13];
lean_object* v___y_174_ = _args[14];
lean_object* v___y_175_ = _args[15];
lean_object* v___y_176_ = _args[16];
_start:
{
lean_object* v_res_177_; 
v_res_177_ = lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___lam__0(v___x_160_, v___x_161_, v_constNames_162_, v___x_163_, v_constants_164_, v_act_165_, v_errorRef_166_, v___x_167_, v_next_168_, v_acc_169_, v_h_170_, v_G_171_, v___y_172_, v___y_173_, v___y_174_, v___y_175_);
lean_dec(v___y_175_);
lean_dec_ref(v___y_174_);
lean_dec(v___y_173_);
lean_dec_ref(v___y_172_);
lean_dec(v_next_168_);
lean_dec(v_errorRef_166_);
lean_dec_ref(v_constants_164_);
lean_dec_ref(v___x_163_);
lean_dec_ref(v_constNames_162_);
lean_dec(v___x_161_);
lean_dec(v___x_160_);
return v_res_177_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___lam__1___closed__0(void){
_start:
{
lean_object* v___x_178_; 
v___x_178_ = l_instMonadEIO(lean_box(0));
return v___x_178_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___lam__1___closed__1(void){
_start:
{
lean_object* v___x_179_; lean_object* v___x_180_; 
v___x_179_ = lean_obj_once(&lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___lam__1___closed__0, &lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___lam__1___closed__0_once, _init_lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___lam__1___closed__0);
v___x_180_ = l_StateRefT_x27_instMonad___redArg(v___x_179_);
return v___x_180_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___lam__1___closed__4(void){
_start:
{
lean_object* v___x_183_; 
v___x_183_ = l_instMonadExceptOfEIO(lean_box(0));
return v___x_183_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___lam__1___closed__5(void){
_start:
{
lean_object* v___x_184_; lean_object* v___f_185_; 
v___x_184_ = lean_obj_once(&lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___lam__1___closed__4, &lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___lam__1___closed__4_once, _init_lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___lam__1___closed__4);
v___f_185_ = lean_alloc_closure((void*)(l_StateRefT_x27_instMonadExceptOf___redArg___lam__0___boxed), 4, 1);
lean_closure_set(v___f_185_, 0, v___x_184_);
return v___f_185_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___lam__1___closed__6(void){
_start:
{
lean_object* v___x_186_; lean_object* v___f_187_; 
v___x_186_ = lean_obj_once(&lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___lam__1___closed__4, &lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___lam__1___closed__4_once, _init_lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___lam__1___closed__4);
v___f_187_ = lean_alloc_closure((void*)(l_StateRefT_x27_instMonadExceptOf___redArg___lam__2), 5, 1);
lean_closure_set(v___f_187_, 0, v___x_186_);
return v___f_187_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___lam__1___closed__7(void){
_start:
{
lean_object* v___f_188_; lean_object* v___f_189_; lean_object* v___x_190_; 
v___f_188_ = lean_obj_once(&lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___lam__1___closed__6, &lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___lam__1___closed__6_once, _init_lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___lam__1___closed__6);
v___f_189_ = lean_obj_once(&lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___lam__1___closed__5, &lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___lam__1___closed__5_once, _init_lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___lam__1___closed__5);
v___x_190_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_190_, 0, v___f_189_);
lean_ctor_set(v___x_190_, 1, v___f_188_);
return v___x_190_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___lam__1___closed__8(void){
_start:
{
lean_object* v___x_191_; lean_object* v___f_192_; 
v___x_191_ = lean_obj_once(&lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___lam__1___closed__7, &lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___lam__1___closed__7_once, _init_lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___lam__1___closed__7);
v___f_192_ = lean_alloc_closure((void*)(l_ReaderT_instMonadExceptOf___redArg___lam__0___boxed), 4, 1);
lean_closure_set(v___f_192_, 0, v___x_191_);
return v___f_192_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___lam__1___closed__9(void){
_start:
{
lean_object* v___x_193_; lean_object* v___f_194_; 
v___x_193_ = lean_obj_once(&lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___lam__1___closed__7, &lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___lam__1___closed__7_once, _init_lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___lam__1___closed__7);
v___f_194_ = lean_alloc_closure((void*)(l_ReaderT_instMonadExceptOf___redArg___lam__2), 5, 1);
lean_closure_set(v___f_194_, 0, v___x_193_);
return v___f_194_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___lam__1___closed__10(void){
_start:
{
lean_object* v___f_195_; lean_object* v___f_196_; lean_object* v___x_197_; 
v___f_195_ = lean_obj_once(&lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___lam__1___closed__9, &lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___lam__1___closed__9_once, _init_lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___lam__1___closed__9);
v___f_196_ = lean_obj_once(&lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___lam__1___closed__8, &lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___lam__1___closed__8_once, _init_lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___lam__1___closed__8);
v___x_197_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_197_, 0, v___f_196_);
lean_ctor_set(v___x_197_, 1, v___f_195_);
return v___x_197_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___lam__1(lean_object* v_stop_198_, lean_object* v_env_199_, lean_object* v___x_200_, lean_object* v___x_201_, lean_object* v___x_202_, lean_object* v___x_203_, lean_object* v_act_204_, lean_object* v_errorRef_205_, lean_object* v___x_206_, lean_object* v_next_207_, lean_object* v_acc_208_, lean_object* v_h_209_, lean_object* v_G_210_, lean_object* v___y_211_, lean_object* v___y_212_, lean_object* v___y_213_, lean_object* v___y_214_){
_start:
{
uint8_t v___x_231_; 
v___x_231_ = lean_nat_dec_lt(v_next_207_, v_stop_198_);
if (v___x_231_ == 0)
{
lean_object* v___x_232_; 
lean_dec_ref(v_G_210_);
lean_dec(v___x_206_);
lean_dec(v_errorRef_205_);
lean_dec_ref(v_act_204_);
lean_dec_ref(v___x_203_);
lean_dec(v___x_202_);
v___x_232_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_232_, 0, v_acc_208_);
return v___x_232_;
}
else
{
lean_object* v___x_233_; lean_object* v_toApplicative_234_; lean_object* v_toFunctor_235_; lean_object* v_toSeq_236_; lean_object* v_toSeqLeft_237_; lean_object* v_toSeqRight_238_; lean_object* v___f_239_; lean_object* v___f_240_; lean_object* v___f_241_; lean_object* v___f_242_; lean_object* v___x_243_; lean_object* v___f_244_; lean_object* v___f_245_; lean_object* v___f_246_; lean_object* v___x_247_; lean_object* v___x_248_; lean_object* v_cancelTk_x3f_249_; 
v___x_233_ = lean_obj_once(&lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___lam__1___closed__1, &lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___lam__1___closed__1_once, _init_lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___lam__1___closed__1);
v_toApplicative_234_ = lean_ctor_get(v___x_233_, 0);
v_toFunctor_235_ = lean_ctor_get(v_toApplicative_234_, 0);
v_toSeq_236_ = lean_ctor_get(v_toApplicative_234_, 2);
v_toSeqLeft_237_ = lean_ctor_get(v_toApplicative_234_, 3);
v_toSeqRight_238_ = lean_ctor_get(v_toApplicative_234_, 4);
v___f_239_ = ((lean_object*)(lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___lam__1___closed__2));
v___f_240_ = ((lean_object*)(lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___lam__1___closed__3));
lean_inc_ref_n(v_toFunctor_235_, 2);
v___f_241_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(v___f_241_, 0, v_toFunctor_235_);
v___f_242_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_242_, 0, v_toFunctor_235_);
v___x_243_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_243_, 0, v___f_241_);
lean_ctor_set(v___x_243_, 1, v___f_242_);
lean_inc(v_toSeqRight_238_);
v___f_244_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_244_, 0, v_toSeqRight_238_);
lean_inc(v_toSeqLeft_237_);
v___f_245_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(v___f_245_, 0, v_toSeqLeft_237_);
lean_inc(v_toSeq_236_);
v___f_246_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(v___f_246_, 0, v_toSeq_236_);
v___x_247_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_247_, 0, v___x_243_);
lean_ctor_set(v___x_247_, 1, v___f_239_);
lean_ctor_set(v___x_247_, 2, v___f_246_);
lean_ctor_set(v___x_247_, 3, v___f_245_);
lean_ctor_set(v___x_247_, 4, v___f_244_);
v___x_248_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_248_, 0, v___x_247_);
lean_ctor_set(v___x_248_, 1, v___f_240_);
v_cancelTk_x3f_249_ = lean_ctor_get(v___y_213_, 12);
if (lean_obj_tag(v_cancelTk_x3f_249_) == 1)
{
lean_object* v_val_250_; uint8_t v___x_251_; 
v_val_250_ = lean_ctor_get(v_cancelTk_x3f_249_, 0);
v___x_251_ = l_IO_CancelToken_isSet(v_val_250_);
if (v___x_251_ == 0)
{
lean_dec_ref_known(v___x_248_, 2);
goto v___jp_216_;
}
else
{
lean_object* v___x_252_; lean_object* v___x_253_; lean_object* v___x_254_; lean_object* v___x_255_; lean_object* v___x_256_; lean_object* v___x_4701__overap_257_; lean_object* v___x_258_; 
v___x_252_ = lean_obj_once(&lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___lam__1___closed__10, &lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___lam__1___closed__10_once, _init_lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___lam__1___closed__10);
v___x_253_ = l_Lean_Core_instMonadRefCoreM;
v___x_254_ = l_Lean_Core_instAddMessageContextCoreM;
v___x_255_ = l_Lean_instAddErrorMessageContextOfAddMessageContextOfMonad___redArg(v___x_254_, v___x_248_);
v___x_256_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_256_, 0, v___x_252_);
lean_ctor_set(v___x_256_, 1, v___x_253_);
lean_ctor_set(v___x_256_, 2, v___x_255_);
v___x_4701__overap_257_ = l_Lean_throwInterruptException___redArg(v___x_256_);
lean_inc(v___y_214_);
lean_inc_ref(v___y_213_);
v___x_258_ = lean_apply_3(v___x_4701__overap_257_, v___y_213_, v___y_214_, lean_box(0));
if (lean_obj_tag(v___x_258_) == 0)
{
lean_dec_ref_known(v___x_258_, 1);
goto v___jp_216_;
}
else
{
lean_object* v_a_259_; lean_object* v___x_261_; uint8_t v_isShared_262_; uint8_t v_isSharedCheck_266_; 
lean_dec_ref(v_G_210_);
lean_dec(v_acc_208_);
lean_dec(v___x_206_);
lean_dec(v_errorRef_205_);
lean_dec_ref(v_act_204_);
lean_dec_ref(v___x_203_);
lean_dec(v___x_202_);
v_a_259_ = lean_ctor_get(v___x_258_, 0);
v_isSharedCheck_266_ = !lean_is_exclusive(v___x_258_);
if (v_isSharedCheck_266_ == 0)
{
v___x_261_ = v___x_258_;
v_isShared_262_ = v_isSharedCheck_266_;
goto v_resetjp_260_;
}
else
{
lean_inc(v_a_259_);
lean_dec(v___x_258_);
v___x_261_ = lean_box(0);
v_isShared_262_ = v_isSharedCheck_266_;
goto v_resetjp_260_;
}
v_resetjp_260_:
{
lean_object* v___x_264_; 
if (v_isShared_262_ == 0)
{
v___x_264_ = v___x_261_;
goto v_reusejp_263_;
}
else
{
lean_object* v_reuseFailAlloc_265_; 
v_reuseFailAlloc_265_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_265_, 0, v_a_259_);
v___x_264_ = v_reuseFailAlloc_265_;
goto v_reusejp_263_;
}
v_reusejp_263_:
{
return v___x_264_;
}
}
}
}
}
else
{
lean_dec_ref_known(v___x_248_, 2);
goto v___jp_216_;
}
}
v___jp_216_:
{
lean_object* v___x_217_; lean_object* v_moduleData_218_; lean_object* v___x_219_; lean_object* v_constNames_220_; lean_object* v_constants_221_; lean_object* v___x_222_; lean_object* v___x_223_; lean_object* v___x_224_; lean_object* v___f_225_; lean_object* v___x_4679__overap_226_; lean_object* v___x_227_; 
v___x_217_ = l_Lean_Environment_header(v_env_199_);
v_moduleData_218_ = lean_ctor_get(v___x_217_, 6);
lean_inc_ref(v_moduleData_218_);
v___x_219_ = lean_array_get(v___x_200_, v_moduleData_218_, v_next_207_);
lean_dec_ref(v_moduleData_218_);
v_constNames_220_ = lean_ctor_get(v___x_219_, 1);
lean_inc_ref(v_constNames_220_);
v_constants_221_ = lean_ctor_get(v___x_219_, 2);
lean_inc_ref(v_constants_221_);
lean_dec(v___x_219_);
v___x_222_ = lean_array_get_size(v_constNames_220_);
v___x_223_ = l_Lean_EnvironmentHeader_moduleNames(v___x_217_);
v___x_224_ = lean_array_get(v___x_201_, v___x_223_, v_next_207_);
lean_dec_ref(v___x_223_);
lean_inc(v___x_202_);
v___f_225_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___lam__0___boxed), 17, 8);
lean_closure_set(v___f_225_, 0, v___x_222_);
lean_closure_set(v___f_225_, 1, v___x_202_);
lean_closure_set(v___f_225_, 2, v_constNames_220_);
lean_closure_set(v___f_225_, 3, v___x_203_);
lean_closure_set(v___f_225_, 4, v_constants_221_);
lean_closure_set(v___f_225_, 5, v_act_204_);
lean_closure_set(v___f_225_, 6, v_errorRef_205_);
lean_closure_set(v___f_225_, 7, v___x_224_);
v___x_4679__overap_226_ = l_WellFounded_opaqueFix_u2083___redArg(v___f_225_, v___x_206_, v_acc_208_, lean_box(0));
lean_inc(v___y_214_);
lean_inc_ref(v___y_213_);
lean_inc(v___y_212_);
lean_inc_ref(v___y_211_);
v___x_227_ = lean_apply_5(v___x_4679__overap_226_, v___y_211_, v___y_212_, v___y_213_, v___y_214_, lean_box(0));
if (lean_obj_tag(v___x_227_) == 0)
{
lean_object* v_a_228_; lean_object* v___x_229_; lean_object* v___x_230_; 
v_a_228_ = lean_ctor_get(v___x_227_, 0);
lean_inc(v_a_228_);
lean_dec_ref_known(v___x_227_, 1);
v___x_229_ = lean_nat_add(v_next_207_, v___x_202_);
lean_dec(v___x_202_);
lean_inc(v___y_214_);
lean_inc_ref(v___y_213_);
lean_inc(v___y_212_);
lean_inc_ref(v___y_211_);
v___x_230_ = lean_apply_9(v_G_210_, v___x_229_, v_a_228_, lean_box(0), lean_box(0), v___y_211_, v___y_212_, v___y_213_, v___y_214_, lean_box(0));
return v___x_230_;
}
else
{
lean_dec_ref(v_G_210_);
lean_dec(v___x_202_);
return v___x_227_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___lam__1___boxed(lean_object** _args){
lean_object* v_stop_267_ = _args[0];
lean_object* v_env_268_ = _args[1];
lean_object* v___x_269_ = _args[2];
lean_object* v___x_270_ = _args[3];
lean_object* v___x_271_ = _args[4];
lean_object* v___x_272_ = _args[5];
lean_object* v_act_273_ = _args[6];
lean_object* v_errorRef_274_ = _args[7];
lean_object* v___x_275_ = _args[8];
lean_object* v_next_276_ = _args[9];
lean_object* v_acc_277_ = _args[10];
lean_object* v_h_278_ = _args[11];
lean_object* v_G_279_ = _args[12];
lean_object* v___y_280_ = _args[13];
lean_object* v___y_281_ = _args[14];
lean_object* v___y_282_ = _args[15];
lean_object* v___y_283_ = _args[16];
lean_object* v___y_284_ = _args[17];
_start:
{
lean_object* v_res_285_; 
v_res_285_ = lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___lam__1(v_stop_267_, v_env_268_, v___x_269_, v___x_270_, v___x_271_, v___x_272_, v_act_273_, v_errorRef_274_, v___x_275_, v_next_276_, v_acc_277_, v_h_278_, v_G_279_, v___y_280_, v___y_281_, v___y_282_, v___y_283_);
lean_dec(v___y_283_);
lean_dec_ref(v___y_282_);
lean_dec(v___y_281_);
lean_dec_ref(v___y_280_);
lean_dec(v_next_276_);
lean_dec(v___x_270_);
lean_dec_ref(v___x_269_);
lean_dec_ref(v_env_268_);
lean_dec(v_stop_267_);
return v_res_285_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___closed__0(void){
_start:
{
lean_object* v___x_286_; 
v___x_286_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_286_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___closed__1(void){
_start:
{
lean_object* v___x_287_; lean_object* v___x_288_; 
v___x_287_ = lean_obj_once(&lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___closed__0, &lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___closed__0_once, _init_lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___closed__0);
v___x_288_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_288_, 0, v___x_287_);
return v___x_288_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___closed__2(void){
_start:
{
lean_object* v___x_289_; lean_object* v___x_290_; lean_object* v___x_291_; 
v___x_289_ = lean_obj_once(&lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___closed__1, &lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___closed__1_once, _init_lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___closed__1);
v___x_290_ = lean_unsigned_to_nat(0u);
v___x_291_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v___x_291_, 0, v___x_290_);
lean_ctor_set(v___x_291_, 1, v___x_290_);
lean_ctor_set(v___x_291_, 2, v___x_290_);
lean_ctor_set(v___x_291_, 3, v___x_290_);
lean_ctor_set(v___x_291_, 4, v___x_289_);
lean_ctor_set(v___x_291_, 5, v___x_289_);
lean_ctor_set(v___x_291_, 6, v___x_289_);
lean_ctor_set(v___x_291_, 7, v___x_289_);
lean_ctor_set(v___x_291_, 8, v___x_289_);
lean_ctor_set(v___x_291_, 9, v___x_289_);
return v___x_291_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___closed__3(void){
_start:
{
lean_object* v___x_292_; lean_object* v___x_293_; 
v___x_292_ = lean_obj_once(&lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___closed__1, &lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___closed__1_once, _init_lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___closed__1);
v___x_293_ = lean_alloc_ctor(0, 6, 0);
lean_ctor_set(v___x_293_, 0, v___x_292_);
lean_ctor_set(v___x_293_, 1, v___x_292_);
lean_ctor_set(v___x_293_, 2, v___x_292_);
lean_ctor_set(v___x_293_, 3, v___x_292_);
lean_ctor_set(v___x_293_, 4, v___x_292_);
lean_ctor_set(v___x_293_, 5, v___x_292_);
return v___x_293_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___closed__4(void){
_start:
{
lean_object* v___x_294_; lean_object* v___x_295_; lean_object* v___x_296_; 
v___x_294_ = lean_unsigned_to_nat(32u);
v___x_295_ = lean_mk_empty_array_with_capacity(v___x_294_);
v___x_296_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_296_, 0, v___x_295_);
return v___x_296_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___closed__5(void){
_start:
{
size_t v___x_297_; lean_object* v___x_298_; lean_object* v___x_299_; lean_object* v___x_300_; lean_object* v___x_301_; lean_object* v___x_302_; 
v___x_297_ = ((size_t)5ULL);
v___x_298_ = lean_unsigned_to_nat(0u);
v___x_299_ = lean_unsigned_to_nat(32u);
v___x_300_ = lean_mk_empty_array_with_capacity(v___x_299_);
v___x_301_ = lean_obj_once(&lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___closed__4, &lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___closed__4_once, _init_lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___closed__4);
v___x_302_ = lean_alloc_ctor(0, 4, sizeof(size_t)*1);
lean_ctor_set(v___x_302_, 0, v___x_301_);
lean_ctor_set(v___x_302_, 1, v___x_300_);
lean_ctor_set(v___x_302_, 2, v___x_298_);
lean_ctor_set(v___x_302_, 3, v___x_298_);
lean_ctor_set_usize(v___x_302_, 4, v___x_297_);
return v___x_302_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___closed__6(void){
_start:
{
lean_object* v___x_303_; lean_object* v___x_304_; 
v___x_303_ = lean_obj_once(&lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___closed__1, &lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___closed__1_once, _init_lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___closed__1);
v___x_304_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_304_, 0, v___x_303_);
lean_ctor_set(v___x_304_, 1, v___x_303_);
lean_ctor_set(v___x_304_, 2, v___x_303_);
lean_ctor_set(v___x_304_, 3, v___x_303_);
lean_ctor_set(v___x_304_, 4, v___x_303_);
return v___x_304_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___closed__7(void){
_start:
{
lean_object* v___x_305_; lean_object* v___x_306_; lean_object* v___x_307_; 
v___x_305_ = lean_unsigned_to_nat(1u);
v___x_306_ = l_Lean_firstFrontendMacroScope;
v___x_307_ = lean_nat_add(v___x_306_, v___x_305_);
return v___x_307_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___closed__9(void){
_start:
{
lean_object* v___x_312_; uint64_t v___x_313_; lean_object* v___x_314_; 
v___x_312_ = lean_obj_once(&lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___closed__5, &lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___closed__5_once, _init_lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___closed__5);
v___x_313_ = 0ULL;
v___x_314_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v___x_314_, 0, v___x_312_);
lean_ctor_set_uint64(v___x_314_, sizeof(void*)*1, v___x_313_);
return v___x_314_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___closed__10(void){
_start:
{
lean_object* v___x_315_; lean_object* v___x_316_; 
v___x_315_ = lean_obj_once(&lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___closed__1, &lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___closed__1_once, _init_lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___closed__1);
v___x_316_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_316_, 0, v___x_315_);
lean_ctor_set(v___x_316_, 1, v___x_315_);
return v___x_316_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___closed__11(void){
_start:
{
lean_object* v___x_317_; lean_object* v___x_318_; lean_object* v___x_319_; 
v___x_317_ = l_Lean_NameSet_empty;
v___x_318_ = lean_obj_once(&lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___closed__5, &lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___closed__5_once, _init_lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___closed__5);
v___x_319_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_319_, 0, v___x_318_);
lean_ctor_set(v___x_319_, 1, v___x_318_);
lean_ctor_set(v___x_319_, 2, v___x_317_);
return v___x_319_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___closed__12(void){
_start:
{
lean_object* v___x_320_; lean_object* v___x_321_; uint8_t v___x_322_; lean_object* v___x_323_; 
v___x_320_ = lean_obj_once(&lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___closed__5, &lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___closed__5_once, _init_lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___closed__5);
v___x_321_ = lean_obj_once(&lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___closed__1, &lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___closed__1_once, _init_lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___closed__1);
v___x_322_ = 1;
v___x_323_ = lean_alloc_ctor(0, 3, 1);
lean_ctor_set(v___x_323_, 0, v___x_321_);
lean_ctor_set(v___x_323_, 1, v___x_321_);
lean_ctor_set(v___x_323_, 2, v___x_320_);
lean_ctor_set_uint8(v___x_323_, sizeof(void*)*3, v___x_322_);
return v___x_323_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___closed__14(void){
_start:
{
lean_object* v___x_326_; lean_object* v___x_327_; lean_object* v___x_328_; lean_object* v___x_329_; lean_object* v___x_330_; lean_object* v___x_331_; 
v___x_326_ = lean_obj_once(&lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___closed__6, &lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___closed__6_once, _init_lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___closed__6);
v___x_327_ = lean_obj_once(&lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___closed__5, &lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___closed__5_once, _init_lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___closed__5);
v___x_328_ = lean_box(1);
v___x_329_ = lean_obj_once(&lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___closed__3, &lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___closed__3_once, _init_lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___closed__3);
v___x_330_ = lean_obj_once(&lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___closed__2, &lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___closed__2_once, _init_lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___closed__2);
v___x_331_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_331_, 0, v___x_330_);
lean_ctor_set(v___x_331_, 1, v___x_329_);
lean_ctor_set(v___x_331_, 2, v___x_328_);
lean_ctor_set(v___x_331_, 3, v___x_327_);
lean_ctor_set(v___x_331_, 4, v___x_326_);
return v___x_331_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg(lean_object* v_ngen_332_, lean_object* v_errorRef_333_, lean_object* v_env_334_, lean_object* v_init_335_, lean_object* v_act_336_, lean_object* v_mctx_337_, lean_object* v_cctx_338_, lean_object* v_start_339_, lean_object* v_stop_340_){
_start:
{
lean_object* v___x_342_; lean_object* v___x_343_; lean_object* v___x_344_; lean_object* v___x_345_; lean_object* v___x_346_; lean_object* v___x_347_; lean_object* v___x_348_; lean_object* v___x_349_; lean_object* v___x_350_; lean_object* v___x_351_; lean_object* v___x_352_; lean_object* v___x_353_; lean_object* v___x_354_; lean_object* v___x_355_; lean_object* v___x_356_; lean_object* v___x_357_; lean_object* v___x_358_; lean_object* v___x_359_; lean_object* v_fileName_360_; lean_object* v_fileMap_361_; lean_object* v_options_362_; lean_object* v_currRecDepth_363_; lean_object* v_ref_364_; lean_object* v_currNamespace_365_; lean_object* v_openDecls_366_; lean_object* v_maxHeartbeats_367_; lean_object* v_quotContext_368_; lean_object* v_currMacroScope_369_; lean_object* v_cancelTk_x3f_370_; uint8_t v_suppressElabErrors_371_; lean_object* v___x_373_; uint8_t v_isShared_374_; uint8_t v_isSharedCheck_428_; 
v___x_342_ = lean_io_get_num_heartbeats();
v___x_343_ = lean_unsigned_to_nat(0u);
v___x_344_ = lean_unsigned_to_nat(1u);
v___x_345_ = lean_obj_once(&lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___closed__7, &lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___closed__7_once, _init_lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___closed__7);
v___x_346_ = lean_box(0);
v___x_347_ = ((lean_object*)(lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___closed__8));
v___x_348_ = lean_obj_once(&lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___closed__9, &lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___closed__9_once, _init_lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___closed__9);
v___x_349_ = lean_obj_once(&lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___closed__10, &lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___closed__10_once, _init_lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___closed__10);
v___x_350_ = lean_obj_once(&lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___closed__11, &lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___closed__11_once, _init_lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___closed__11);
v___x_351_ = lean_obj_once(&lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___closed__12, &lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___closed__12_once, _init_lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___closed__12);
v___x_352_ = ((lean_object*)(lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___closed__13));
lean_inc_ref(v_env_334_);
v___x_353_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v___x_353_, 0, v_env_334_);
lean_ctor_set(v___x_353_, 1, v___x_345_);
lean_ctor_set(v___x_353_, 2, v_ngen_332_);
lean_ctor_set(v___x_353_, 3, v___x_347_);
lean_ctor_set(v___x_353_, 4, v___x_348_);
lean_ctor_set(v___x_353_, 5, v___x_349_);
lean_ctor_set(v___x_353_, 6, v___x_350_);
lean_ctor_set(v___x_353_, 7, v___x_351_);
lean_ctor_set(v___x_353_, 8, v___x_352_);
v___x_354_ = lean_st_mk_ref(v___x_353_);
v___x_355_ = l_Lean_inheritedTraceOptions;
v___x_356_ = lean_st_ref_get(v___x_355_);
v___x_357_ = l_Lean_KVMap_instValueBool;
v___x_358_ = l_Lean_KVMap_instValueNat;
v___x_359_ = lean_st_ref_get(v___x_354_);
v_fileName_360_ = lean_ctor_get(v_cctx_338_, 0);
v_fileMap_361_ = lean_ctor_get(v_cctx_338_, 1);
v_options_362_ = lean_ctor_get(v_cctx_338_, 2);
v_currRecDepth_363_ = lean_ctor_get(v_cctx_338_, 3);
v_ref_364_ = lean_ctor_get(v_cctx_338_, 5);
v_currNamespace_365_ = lean_ctor_get(v_cctx_338_, 6);
v_openDecls_366_ = lean_ctor_get(v_cctx_338_, 7);
v_maxHeartbeats_367_ = lean_ctor_get(v_cctx_338_, 9);
v_quotContext_368_ = lean_ctor_get(v_cctx_338_, 10);
v_currMacroScope_369_ = lean_ctor_get(v_cctx_338_, 11);
v_cancelTk_x3f_370_ = lean_ctor_get(v_cctx_338_, 12);
v_suppressElabErrors_371_ = lean_ctor_get_uint8(v_cctx_338_, sizeof(void*)*14 + 1);
v_isSharedCheck_428_ = !lean_is_exclusive(v_cctx_338_);
if (v_isSharedCheck_428_ == 0)
{
lean_object* v_unused_429_; lean_object* v_unused_430_; lean_object* v_unused_431_; 
v_unused_429_ = lean_ctor_get(v_cctx_338_, 13);
lean_dec(v_unused_429_);
v_unused_430_ = lean_ctor_get(v_cctx_338_, 8);
lean_dec(v_unused_430_);
v_unused_431_ = lean_ctor_get(v_cctx_338_, 4);
lean_dec(v_unused_431_);
v___x_373_ = v_cctx_338_;
v_isShared_374_ = v_isSharedCheck_428_;
goto v_resetjp_372_;
}
else
{
lean_inc(v_cancelTk_x3f_370_);
lean_inc(v_currMacroScope_369_);
lean_inc(v_quotContext_368_);
lean_inc(v_maxHeartbeats_367_);
lean_inc(v_openDecls_366_);
lean_inc(v_currNamespace_365_);
lean_inc(v_ref_364_);
lean_inc(v_currRecDepth_363_);
lean_inc(v_options_362_);
lean_inc(v_fileMap_361_);
lean_inc(v_fileName_360_);
lean_dec(v_cctx_338_);
v___x_373_ = lean_box(0);
v_isShared_374_ = v_isSharedCheck_428_;
goto v_resetjp_372_;
}
v_resetjp_372_:
{
lean_object* v_env_375_; lean_object* v___x_376_; lean_object* v___x_377_; lean_object* v___f_378_; lean_object* v___x_379_; lean_object* v___x_380_; lean_object* v___x_381_; lean_object* v___y_383_; uint8_t v___y_404_; uint8_t v___x_425_; 
v_env_375_ = lean_ctor_get(v___x_359_, 0);
lean_inc_ref(v_env_375_);
lean_dec(v___x_359_);
v___x_376_ = l_Lean_instInhabitedConstantInfo_default;
v___x_377_ = l_Lean_instInhabitedModuleData_default;
v___f_378_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___lam__1___boxed), 18, 9);
lean_closure_set(v___f_378_, 0, v_stop_340_);
lean_closure_set(v___f_378_, 1, v_env_334_);
lean_closure_set(v___f_378_, 2, v___x_377_);
lean_closure_set(v___f_378_, 3, v___x_346_);
lean_closure_set(v___f_378_, 4, v___x_344_);
lean_closure_set(v___f_378_, 5, v___x_376_);
lean_closure_set(v___f_378_, 6, v_act_336_);
lean_closure_set(v___f_378_, 7, v_errorRef_333_);
lean_closure_set(v___f_378_, 8, v___x_343_);
v___x_379_ = lean_obj_once(&lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___closed__14, &lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___closed__14_once, _init_lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___closed__14);
v___x_380_ = l_Lean_diagnostics;
v___x_381_ = l_Lean_Option_get___redArg(v___x_357_, v_options_362_, v___x_380_);
v___x_425_ = l_Lean_Kernel_isDiagnosticsEnabled(v_env_375_);
lean_dec_ref(v_env_375_);
if (v___x_425_ == 0)
{
uint8_t v___x_426_; 
v___x_426_ = lean_unbox(v___x_381_);
if (v___x_426_ == 0)
{
lean_inc(v___x_354_);
v___y_383_ = v___x_354_;
goto v___jp_382_;
}
else
{
v___y_404_ = v___x_425_;
goto v___jp_403_;
}
}
else
{
uint8_t v___x_427_; 
v___x_427_ = lean_unbox(v___x_381_);
v___y_404_ = v___x_427_;
goto v___jp_403_;
}
v___jp_382_:
{
lean_object* v___x_384_; lean_object* v___x_385_; lean_object* v___x_386_; lean_object* v___x_388_; 
v___x_384_ = lean_st_mk_ref(v___x_379_);
v___x_385_ = l_Lean_maxRecDepth;
v___x_386_ = l_Lean_Option_get___redArg(v___x_358_, v_options_362_, v___x_385_);
if (v_isShared_374_ == 0)
{
lean_ctor_set(v___x_373_, 13, v___x_356_);
lean_ctor_set(v___x_373_, 8, v___x_342_);
lean_ctor_set(v___x_373_, 4, v___x_386_);
v___x_388_ = v___x_373_;
goto v_reusejp_387_;
}
else
{
lean_object* v_reuseFailAlloc_402_; 
v_reuseFailAlloc_402_ = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(v_reuseFailAlloc_402_, 0, v_fileName_360_);
lean_ctor_set(v_reuseFailAlloc_402_, 1, v_fileMap_361_);
lean_ctor_set(v_reuseFailAlloc_402_, 2, v_options_362_);
lean_ctor_set(v_reuseFailAlloc_402_, 3, v_currRecDepth_363_);
lean_ctor_set(v_reuseFailAlloc_402_, 4, v___x_386_);
lean_ctor_set(v_reuseFailAlloc_402_, 5, v_ref_364_);
lean_ctor_set(v_reuseFailAlloc_402_, 6, v_currNamespace_365_);
lean_ctor_set(v_reuseFailAlloc_402_, 7, v_openDecls_366_);
lean_ctor_set(v_reuseFailAlloc_402_, 8, v___x_342_);
lean_ctor_set(v_reuseFailAlloc_402_, 9, v_maxHeartbeats_367_);
lean_ctor_set(v_reuseFailAlloc_402_, 10, v_quotContext_368_);
lean_ctor_set(v_reuseFailAlloc_402_, 11, v_currMacroScope_369_);
lean_ctor_set(v_reuseFailAlloc_402_, 12, v_cancelTk_x3f_370_);
lean_ctor_set(v_reuseFailAlloc_402_, 13, v___x_356_);
v___x_388_ = v_reuseFailAlloc_402_;
goto v_reusejp_387_;
}
v_reusejp_387_:
{
uint8_t v___x_389_; lean_object* v___x_4234__overap_390_; lean_object* v___x_391_; 
v___x_389_ = lean_unbox(v___x_381_);
lean_dec(v___x_381_);
lean_ctor_set_uint8(v___x_388_, sizeof(void*)*14, v___x_389_);
lean_ctor_set_uint8(v___x_388_, sizeof(void*)*14 + 1, v_suppressElabErrors_371_);
v___x_4234__overap_390_ = l_WellFounded_opaqueFix_u2083___redArg(v___f_378_, v_start_339_, v_init_335_, lean_box(0));
lean_inc(v___x_384_);
v___x_391_ = lean_apply_5(v___x_4234__overap_390_, v_mctx_337_, v___x_384_, v___x_388_, v___y_383_, lean_box(0));
if (lean_obj_tag(v___x_391_) == 0)
{
lean_object* v_a_392_; lean_object* v___x_394_; uint8_t v_isShared_395_; uint8_t v_isSharedCheck_401_; 
v_a_392_ = lean_ctor_get(v___x_391_, 0);
v_isSharedCheck_401_ = !lean_is_exclusive(v___x_391_);
if (v_isSharedCheck_401_ == 0)
{
v___x_394_ = v___x_391_;
v_isShared_395_ = v_isSharedCheck_401_;
goto v_resetjp_393_;
}
else
{
lean_inc(v_a_392_);
lean_dec(v___x_391_);
v___x_394_ = lean_box(0);
v_isShared_395_ = v_isSharedCheck_401_;
goto v_resetjp_393_;
}
v_resetjp_393_:
{
lean_object* v___x_396_; lean_object* v___x_397_; lean_object* v___x_399_; 
v___x_396_ = lean_st_ref_get(v___x_384_);
lean_dec(v___x_384_);
lean_dec(v___x_396_);
v___x_397_ = lean_st_ref_get(v___x_354_);
lean_dec(v___x_354_);
lean_dec(v___x_397_);
if (v_isShared_395_ == 0)
{
v___x_399_ = v___x_394_;
goto v_reusejp_398_;
}
else
{
lean_object* v_reuseFailAlloc_400_; 
v_reuseFailAlloc_400_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_400_, 0, v_a_392_);
v___x_399_ = v_reuseFailAlloc_400_;
goto v_reusejp_398_;
}
v_reusejp_398_:
{
return v___x_399_;
}
}
}
else
{
lean_dec(v___x_384_);
lean_dec(v___x_354_);
return v___x_391_;
}
}
}
v___jp_403_:
{
if (v___y_404_ == 0)
{
lean_object* v___x_405_; lean_object* v_env_406_; lean_object* v_nextMacroScope_407_; lean_object* v_ngen_408_; lean_object* v_auxDeclNGen_409_; lean_object* v_traceState_410_; lean_object* v_messages_411_; lean_object* v_infoState_412_; lean_object* v_snapshotTasks_413_; lean_object* v___x_415_; uint8_t v_isShared_416_; uint8_t v_isSharedCheck_423_; 
v___x_405_ = lean_st_ref_take(v___x_354_);
v_env_406_ = lean_ctor_get(v___x_405_, 0);
v_nextMacroScope_407_ = lean_ctor_get(v___x_405_, 1);
v_ngen_408_ = lean_ctor_get(v___x_405_, 2);
v_auxDeclNGen_409_ = lean_ctor_get(v___x_405_, 3);
v_traceState_410_ = lean_ctor_get(v___x_405_, 4);
v_messages_411_ = lean_ctor_get(v___x_405_, 6);
v_infoState_412_ = lean_ctor_get(v___x_405_, 7);
v_snapshotTasks_413_ = lean_ctor_get(v___x_405_, 8);
v_isSharedCheck_423_ = !lean_is_exclusive(v___x_405_);
if (v_isSharedCheck_423_ == 0)
{
lean_object* v_unused_424_; 
v_unused_424_ = lean_ctor_get(v___x_405_, 5);
lean_dec(v_unused_424_);
v___x_415_ = v___x_405_;
v_isShared_416_ = v_isSharedCheck_423_;
goto v_resetjp_414_;
}
else
{
lean_inc(v_snapshotTasks_413_);
lean_inc(v_infoState_412_);
lean_inc(v_messages_411_);
lean_inc(v_traceState_410_);
lean_inc(v_auxDeclNGen_409_);
lean_inc(v_ngen_408_);
lean_inc(v_nextMacroScope_407_);
lean_inc(v_env_406_);
lean_dec(v___x_405_);
v___x_415_ = lean_box(0);
v_isShared_416_ = v_isSharedCheck_423_;
goto v_resetjp_414_;
}
v_resetjp_414_:
{
uint8_t v___x_417_; lean_object* v___x_418_; lean_object* v___x_420_; 
v___x_417_ = lean_unbox(v___x_381_);
v___x_418_ = l_Lean_Kernel_enableDiag(v_env_406_, v___x_417_);
if (v_isShared_416_ == 0)
{
lean_ctor_set(v___x_415_, 5, v___x_349_);
lean_ctor_set(v___x_415_, 0, v___x_418_);
v___x_420_ = v___x_415_;
goto v_reusejp_419_;
}
else
{
lean_object* v_reuseFailAlloc_422_; 
v_reuseFailAlloc_422_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_422_, 0, v___x_418_);
lean_ctor_set(v_reuseFailAlloc_422_, 1, v_nextMacroScope_407_);
lean_ctor_set(v_reuseFailAlloc_422_, 2, v_ngen_408_);
lean_ctor_set(v_reuseFailAlloc_422_, 3, v_auxDeclNGen_409_);
lean_ctor_set(v_reuseFailAlloc_422_, 4, v_traceState_410_);
lean_ctor_set(v_reuseFailAlloc_422_, 5, v___x_349_);
lean_ctor_set(v_reuseFailAlloc_422_, 6, v_messages_411_);
lean_ctor_set(v_reuseFailAlloc_422_, 7, v_infoState_412_);
lean_ctor_set(v_reuseFailAlloc_422_, 8, v_snapshotTasks_413_);
v___x_420_ = v_reuseFailAlloc_422_;
goto v_reusejp_419_;
}
v_reusejp_419_:
{
lean_object* v___x_421_; 
v___x_421_ = lean_st_ref_set(v___x_354_, v___x_420_);
lean_inc(v___x_354_);
v___y_383_ = v___x_354_;
goto v___jp_382_;
}
}
}
else
{
lean_inc(v___x_354_);
v___y_383_ = v___x_354_;
goto v___jp_382_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___boxed(lean_object* v_ngen_432_, lean_object* v_errorRef_433_, lean_object* v_env_434_, lean_object* v_init_435_, lean_object* v_act_436_, lean_object* v_mctx_437_, lean_object* v_cctx_438_, lean_object* v_start_439_, lean_object* v_stop_440_, lean_object* v_a_441_){
_start:
{
lean_object* v_res_442_; 
v_res_442_ = lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg(v_ngen_432_, v_errorRef_433_, v_env_434_, v_init_435_, v_act_436_, v_mctx_437_, v_cctx_438_, v_start_439_, v_stop_440_);
return v_res_442_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules(lean_object* v_00_u03b1_443_, lean_object* v_ngen_444_, lean_object* v_errorRef_445_, lean_object* v_env_446_, lean_object* v_init_447_, lean_object* v_act_448_, lean_object* v_mctx_449_, lean_object* v_cctx_450_, lean_object* v_start_451_, lean_object* v_stop_452_){
_start:
{
lean_object* v___x_454_; 
v___x_454_ = lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg(v_ngen_444_, v_errorRef_445_, v_env_446_, v_init_447_, v_act_448_, v_mctx_449_, v_cctx_450_, v_start_451_, v_stop_452_);
return v___x_454_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___boxed(lean_object* v_00_u03b1_455_, lean_object* v_ngen_456_, lean_object* v_errorRef_457_, lean_object* v_env_458_, lean_object* v_init_459_, lean_object* v_act_460_, lean_object* v_mctx_461_, lean_object* v_cctx_462_, lean_object* v_start_463_, lean_object* v_stop_464_, lean_object* v_a_465_){
_start:
{
lean_object* v_res_466_; 
v_res_466_ = lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules(v_00_u03b1_455_, v_ngen_456_, v_errorRef_457_, v_env_458_, v_init_459_, v_act_460_, v_mctx_461_, v_cctx_462_, v_start_463_, v_stop_464_);
return v_res_466_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_foldImportedDecls___redArg___lam__0(lean_object* v___x_467_, lean_object* v_moduleData_468_, lean_object* v_constantsPerTask_469_, lean_object* v_val_470_, lean_object* v_env_471_, lean_object* v_init_472_, lean_object* v_act_473_, lean_object* v___x_474_, lean_object* v_a_475_, lean_object* v___x_476_, lean_object* v_next_477_, lean_object* v_acc_478_, lean_object* v_h_479_, lean_object* v_G_480_, lean_object* v___y_481_, lean_object* v___y_482_){
_start:
{
lean_object* v_a_485_; uint8_t v___x_489_; 
v___x_489_ = lean_nat_dec_lt(v_next_477_, v___x_467_);
if (v___x_489_ == 0)
{
lean_object* v___x_490_; 
lean_dec_ref(v_G_480_);
lean_dec(v___x_476_);
lean_dec_ref(v___x_474_);
lean_dec_ref(v_act_473_);
lean_dec(v_init_472_);
lean_dec_ref(v_env_471_);
lean_dec(v_val_470_);
v___x_490_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_490_, 0, v_acc_478_);
return v___x_490_;
}
else
{
lean_object* v_snd_491_; lean_object* v_fst_492_; lean_object* v___x_494_; uint8_t v_isShared_495_; uint8_t v_isSharedCheck_557_; 
v_snd_491_ = lean_ctor_get(v_acc_478_, 1);
v_fst_492_ = lean_ctor_get(v_acc_478_, 0);
v_isSharedCheck_557_ = !lean_is_exclusive(v_acc_478_);
if (v_isSharedCheck_557_ == 0)
{
v___x_494_ = v_acc_478_;
v_isShared_495_ = v_isSharedCheck_557_;
goto v_resetjp_493_;
}
else
{
lean_inc(v_snd_491_);
lean_inc(v_fst_492_);
lean_dec(v_acc_478_);
v___x_494_ = lean_box(0);
v_isShared_495_ = v_isSharedCheck_557_;
goto v_resetjp_493_;
}
v_resetjp_493_:
{
lean_object* v_fst_496_; lean_object* v_snd_497_; lean_object* v___x_499_; uint8_t v_isShared_500_; uint8_t v_isSharedCheck_556_; 
v_fst_496_ = lean_ctor_get(v_snd_491_, 0);
v_snd_497_ = lean_ctor_get(v_snd_491_, 1);
v_isSharedCheck_556_ = !lean_is_exclusive(v_snd_491_);
if (v_isSharedCheck_556_ == 0)
{
v___x_499_ = v_snd_491_;
v_isShared_500_ = v_isSharedCheck_556_;
goto v_resetjp_498_;
}
else
{
lean_inc(v_snd_497_);
lean_inc(v_fst_496_);
lean_dec(v_snd_491_);
v___x_499_ = lean_box(0);
v_isShared_500_ = v_isSharedCheck_556_;
goto v_resetjp_498_;
}
v_resetjp_498_:
{
lean_object* v___x_501_; lean_object* v_constants_502_; lean_object* v___x_503_; lean_object* v___x_504_; uint8_t v___x_505_; 
v___x_501_ = lean_array_fget_borrowed(v_moduleData_468_, v_next_477_);
v_constants_502_ = lean_ctor_get(v___x_501_, 2);
v___x_503_ = lean_array_get_size(v_constants_502_);
v___x_504_ = lean_nat_add(v_snd_497_, v___x_503_);
lean_dec(v_snd_497_);
v___x_505_ = lean_nat_dec_lt(v_constantsPerTask_469_, v___x_504_);
if (v___x_505_ == 0)
{
lean_object* v___x_507_; 
lean_dec(v___x_476_);
lean_dec_ref(v___x_474_);
lean_dec_ref(v_act_473_);
lean_dec(v_init_472_);
lean_dec_ref(v_env_471_);
lean_dec(v_val_470_);
if (v_isShared_500_ == 0)
{
lean_ctor_set(v___x_499_, 1, v___x_504_);
v___x_507_ = v___x_499_;
goto v_reusejp_506_;
}
else
{
lean_object* v_reuseFailAlloc_511_; 
v_reuseFailAlloc_511_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_511_, 0, v_fst_496_);
lean_ctor_set(v_reuseFailAlloc_511_, 1, v___x_504_);
v___x_507_ = v_reuseFailAlloc_511_;
goto v_reusejp_506_;
}
v_reusejp_506_:
{
lean_object* v___x_509_; 
if (v_isShared_495_ == 0)
{
lean_ctor_set(v___x_494_, 1, v___x_507_);
v___x_509_ = v___x_494_;
goto v_reusejp_508_;
}
else
{
lean_object* v_reuseFailAlloc_510_; 
v_reuseFailAlloc_510_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_510_, 0, v_fst_492_);
lean_ctor_set(v_reuseFailAlloc_510_, 1, v___x_507_);
v___x_509_ = v_reuseFailAlloc_510_;
goto v_reusejp_508_;
}
v_reusejp_508_:
{
v_a_485_ = v___x_509_;
goto v___jp_484_;
}
}
}
else
{
lean_object* v___x_512_; lean_object* v___x_513_; lean_object* v_ngen_514_; lean_object* v_namePrefix_515_; lean_object* v_idx_516_; lean_object* v___x_518_; uint8_t v_isShared_519_; uint8_t v_isSharedCheck_555_; 
lean_dec(v___x_504_);
v___x_512_ = lean_st_ref_get(v___y_482_);
v___x_513_ = lean_st_ref_take(v___y_482_);
v_ngen_514_ = lean_ctor_get(v___x_512_, 2);
lean_inc_ref(v_ngen_514_);
lean_dec(v___x_512_);
v_namePrefix_515_ = lean_ctor_get(v_ngen_514_, 0);
v_idx_516_ = lean_ctor_get(v_ngen_514_, 1);
v_isSharedCheck_555_ = !lean_is_exclusive(v_ngen_514_);
if (v_isSharedCheck_555_ == 0)
{
v___x_518_ = v_ngen_514_;
v_isShared_519_ = v_isSharedCheck_555_;
goto v_resetjp_517_;
}
else
{
lean_inc(v_idx_516_);
lean_inc(v_namePrefix_515_);
lean_dec(v_ngen_514_);
v___x_518_ = lean_box(0);
v_isShared_519_ = v_isSharedCheck_555_;
goto v_resetjp_517_;
}
v_resetjp_517_:
{
lean_object* v_env_520_; lean_object* v_nextMacroScope_521_; lean_object* v_auxDeclNGen_522_; lean_object* v_traceState_523_; lean_object* v_cache_524_; lean_object* v_messages_525_; lean_object* v_infoState_526_; lean_object* v_snapshotTasks_527_; lean_object* v___x_529_; uint8_t v_isShared_530_; uint8_t v_isSharedCheck_553_; 
v_env_520_ = lean_ctor_get(v___x_513_, 0);
v_nextMacroScope_521_ = lean_ctor_get(v___x_513_, 1);
v_auxDeclNGen_522_ = lean_ctor_get(v___x_513_, 3);
v_traceState_523_ = lean_ctor_get(v___x_513_, 4);
v_cache_524_ = lean_ctor_get(v___x_513_, 5);
v_messages_525_ = lean_ctor_get(v___x_513_, 6);
v_infoState_526_ = lean_ctor_get(v___x_513_, 7);
v_snapshotTasks_527_ = lean_ctor_get(v___x_513_, 8);
v_isSharedCheck_553_ = !lean_is_exclusive(v___x_513_);
if (v_isSharedCheck_553_ == 0)
{
lean_object* v_unused_554_; 
v_unused_554_ = lean_ctor_get(v___x_513_, 2);
lean_dec(v_unused_554_);
v___x_529_ = v___x_513_;
v_isShared_530_ = v_isSharedCheck_553_;
goto v_resetjp_528_;
}
else
{
lean_inc(v_snapshotTasks_527_);
lean_inc(v_infoState_526_);
lean_inc(v_messages_525_);
lean_inc(v_cache_524_);
lean_inc(v_traceState_523_);
lean_inc(v_auxDeclNGen_522_);
lean_inc(v_nextMacroScope_521_);
lean_inc(v_env_520_);
lean_dec(v___x_513_);
v___x_529_ = lean_box(0);
v_isShared_530_ = v_isSharedCheck_553_;
goto v_resetjp_528_;
}
v_resetjp_528_:
{
lean_object* v___x_531_; lean_object* v___x_532_; lean_object* v___x_533_; lean_object* v___x_535_; 
lean_inc(v_idx_516_);
lean_inc(v_namePrefix_515_);
v___x_531_ = l_Lean_Name_num___override(v_namePrefix_515_, v_idx_516_);
v___x_532_ = lean_unsigned_to_nat(1u);
v___x_533_ = lean_nat_add(v_idx_516_, v___x_532_);
lean_dec(v_idx_516_);
if (v_isShared_519_ == 0)
{
lean_ctor_set(v___x_518_, 1, v___x_533_);
v___x_535_ = v___x_518_;
goto v_reusejp_534_;
}
else
{
lean_object* v_reuseFailAlloc_552_; 
v_reuseFailAlloc_552_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_552_, 0, v_namePrefix_515_);
lean_ctor_set(v_reuseFailAlloc_552_, 1, v___x_533_);
v___x_535_ = v_reuseFailAlloc_552_;
goto v_reusejp_534_;
}
v_reusejp_534_:
{
lean_object* v___x_537_; 
if (v_isShared_530_ == 0)
{
lean_ctor_set(v___x_529_, 2, v___x_535_);
v___x_537_ = v___x_529_;
goto v_reusejp_536_;
}
else
{
lean_object* v_reuseFailAlloc_551_; 
v_reuseFailAlloc_551_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_551_, 0, v_env_520_);
lean_ctor_set(v_reuseFailAlloc_551_, 1, v_nextMacroScope_521_);
lean_ctor_set(v_reuseFailAlloc_551_, 2, v___x_535_);
lean_ctor_set(v_reuseFailAlloc_551_, 3, v_auxDeclNGen_522_);
lean_ctor_set(v_reuseFailAlloc_551_, 4, v_traceState_523_);
lean_ctor_set(v_reuseFailAlloc_551_, 5, v_cache_524_);
lean_ctor_set(v_reuseFailAlloc_551_, 6, v_messages_525_);
lean_ctor_set(v_reuseFailAlloc_551_, 7, v_infoState_526_);
lean_ctor_set(v_reuseFailAlloc_551_, 8, v_snapshotTasks_527_);
v___x_537_ = v_reuseFailAlloc_551_;
goto v_reusejp_536_;
}
v_reusejp_536_:
{
lean_object* v___x_538_; lean_object* v___x_539_; lean_object* v___x_540_; lean_object* v___x_541_; lean_object* v___x_542_; lean_object* v___x_543_; lean_object* v___x_544_; lean_object* v___x_546_; 
v___x_538_ = lean_st_ref_set(v___y_482_, v___x_537_);
v___x_539_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_539_, 0, v___x_531_);
lean_ctor_set(v___x_539_, 1, v___x_532_);
v___x_540_ = lean_nat_add(v_next_477_, v___x_532_);
lean_inc(v___x_540_);
lean_inc_ref(v_a_475_);
v___x_541_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___boxed), 11, 10);
lean_closure_set(v___x_541_, 0, lean_box(0));
lean_closure_set(v___x_541_, 1, v___x_539_);
lean_closure_set(v___x_541_, 2, v_val_470_);
lean_closure_set(v___x_541_, 3, v_env_471_);
lean_closure_set(v___x_541_, 4, v_init_472_);
lean_closure_set(v___x_541_, 5, v_act_473_);
lean_closure_set(v___x_541_, 6, v___x_474_);
lean_closure_set(v___x_541_, 7, v_a_475_);
lean_closure_set(v___x_541_, 8, v_fst_496_);
lean_closure_set(v___x_541_, 9, v___x_540_);
v___x_542_ = lean_alloc_closure((void*)(l_EIO_toBaseIO___boxed), 4, 3);
lean_closure_set(v___x_542_, 0, lean_box(0));
lean_closure_set(v___x_542_, 1, lean_box(0));
lean_closure_set(v___x_542_, 2, v___x_541_);
lean_inc(v___x_476_);
v___x_543_ = lean_io_as_task(v___x_542_, v___x_476_);
v___x_544_ = lean_array_push(v_fst_492_, v___x_543_);
if (v_isShared_500_ == 0)
{
lean_ctor_set(v___x_499_, 1, v___x_476_);
lean_ctor_set(v___x_499_, 0, v___x_540_);
v___x_546_ = v___x_499_;
goto v_reusejp_545_;
}
else
{
lean_object* v_reuseFailAlloc_550_; 
v_reuseFailAlloc_550_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_550_, 0, v___x_540_);
lean_ctor_set(v_reuseFailAlloc_550_, 1, v___x_476_);
v___x_546_ = v_reuseFailAlloc_550_;
goto v_reusejp_545_;
}
v_reusejp_545_:
{
lean_object* v___x_548_; 
if (v_isShared_495_ == 0)
{
lean_ctor_set(v___x_494_, 1, v___x_546_);
lean_ctor_set(v___x_494_, 0, v___x_544_);
v___x_548_ = v___x_494_;
goto v_reusejp_547_;
}
else
{
lean_object* v_reuseFailAlloc_549_; 
v_reuseFailAlloc_549_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_549_, 0, v___x_544_);
lean_ctor_set(v_reuseFailAlloc_549_, 1, v___x_546_);
v___x_548_ = v_reuseFailAlloc_549_;
goto v_reusejp_547_;
}
v_reusejp_547_:
{
v_a_485_ = v___x_548_;
goto v___jp_484_;
}
}
}
}
}
}
}
}
}
}
v___jp_484_:
{
lean_object* v___x_486_; lean_object* v___x_487_; lean_object* v___x_488_; 
v___x_486_ = lean_unsigned_to_nat(1u);
v___x_487_ = lean_nat_add(v_next_477_, v___x_486_);
lean_inc(v___y_482_);
lean_inc_ref(v___y_481_);
v___x_488_ = lean_apply_7(v_G_480_, v___x_487_, v_a_485_, lean_box(0), lean_box(0), v___y_481_, v___y_482_, lean_box(0));
return v___x_488_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_foldImportedDecls___redArg___lam__0___boxed(lean_object** _args){
lean_object* v___x_558_ = _args[0];
lean_object* v_moduleData_559_ = _args[1];
lean_object* v_constantsPerTask_560_ = _args[2];
lean_object* v_val_561_ = _args[3];
lean_object* v_env_562_ = _args[4];
lean_object* v_init_563_ = _args[5];
lean_object* v_act_564_ = _args[6];
lean_object* v___x_565_ = _args[7];
lean_object* v_a_566_ = _args[8];
lean_object* v___x_567_ = _args[9];
lean_object* v_next_568_ = _args[10];
lean_object* v_acc_569_ = _args[11];
lean_object* v_h_570_ = _args[12];
lean_object* v_G_571_ = _args[13];
lean_object* v___y_572_ = _args[14];
lean_object* v___y_573_ = _args[15];
lean_object* v___y_574_ = _args[16];
_start:
{
lean_object* v_res_575_; 
v_res_575_ = lp_mathlib_Lean_Meta_foldImportedDecls___redArg___lam__0(v___x_558_, v_moduleData_559_, v_constantsPerTask_560_, v_val_561_, v_env_562_, v_init_563_, v_act_564_, v___x_565_, v_a_566_, v___x_567_, v_next_568_, v_acc_569_, v_h_570_, v_G_571_, v___y_572_, v___y_573_);
lean_dec(v___y_573_);
lean_dec_ref(v___y_572_);
lean_dec(v_next_568_);
lean_dec_ref(v_a_566_);
lean_dec(v_constantsPerTask_560_);
lean_dec_ref(v_moduleData_559_);
lean_dec(v___x_558_);
return v_res_575_;
}
}
static lean_object* _init_lp_mathlib_Lean_Meta_foldImportedDecls___redArg___closed__0(void){
_start:
{
lean_object* v___x_576_; 
v___x_576_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_576_;
}
}
static lean_object* _init_lp_mathlib_Lean_Meta_foldImportedDecls___redArg___closed__1(void){
_start:
{
lean_object* v___x_577_; lean_object* v___x_578_; 
v___x_577_ = lean_obj_once(&lp_mathlib_Lean_Meta_foldImportedDecls___redArg___closed__0, &lp_mathlib_Lean_Meta_foldImportedDecls___redArg___closed__0_once, _init_lp_mathlib_Lean_Meta_foldImportedDecls___redArg___closed__0);
v___x_578_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_578_, 0, v___x_577_);
return v___x_578_;
}
}
static lean_object* _init_lp_mathlib_Lean_Meta_foldImportedDecls___redArg___closed__2(void){
_start:
{
lean_object* v___x_579_; lean_object* v___x_580_; lean_object* v___x_581_; 
v___x_579_ = lean_unsigned_to_nat(32u);
v___x_580_ = lean_mk_empty_array_with_capacity(v___x_579_);
v___x_581_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_581_, 0, v___x_580_);
return v___x_581_;
}
}
static lean_object* _init_lp_mathlib_Lean_Meta_foldImportedDecls___redArg___closed__3(void){
_start:
{
size_t v___x_582_; lean_object* v___x_583_; lean_object* v___x_584_; lean_object* v___x_585_; lean_object* v___x_586_; lean_object* v___x_587_; 
v___x_582_ = ((size_t)5ULL);
v___x_583_ = lean_unsigned_to_nat(0u);
v___x_584_ = lean_unsigned_to_nat(32u);
v___x_585_ = lean_mk_empty_array_with_capacity(v___x_584_);
v___x_586_ = lean_obj_once(&lp_mathlib_Lean_Meta_foldImportedDecls___redArg___closed__2, &lp_mathlib_Lean_Meta_foldImportedDecls___redArg___closed__2_once, _init_lp_mathlib_Lean_Meta_foldImportedDecls___redArg___closed__2);
v___x_587_ = lean_alloc_ctor(0, 4, sizeof(size_t)*1);
lean_ctor_set(v___x_587_, 0, v___x_586_);
lean_ctor_set(v___x_587_, 1, v___x_585_);
lean_ctor_set(v___x_587_, 2, v___x_583_);
lean_ctor_set(v___x_587_, 3, v___x_583_);
lean_ctor_set_usize(v___x_587_, 4, v___x_582_);
return v___x_587_;
}
}
static lean_object* _init_lp_mathlib_Lean_Meta_foldImportedDecls___redArg___closed__4(void){
_start:
{
lean_object* v___x_588_; lean_object* v___x_589_; lean_object* v___x_590_; lean_object* v___x_591_; 
v___x_588_ = lean_box(1);
v___x_589_ = lean_obj_once(&lp_mathlib_Lean_Meta_foldImportedDecls___redArg___closed__3, &lp_mathlib_Lean_Meta_foldImportedDecls___redArg___closed__3_once, _init_lp_mathlib_Lean_Meta_foldImportedDecls___redArg___closed__3);
v___x_590_ = lean_obj_once(&lp_mathlib_Lean_Meta_foldImportedDecls___redArg___closed__1, &lp_mathlib_Lean_Meta_foldImportedDecls___redArg___closed__1_once, _init_lp_mathlib_Lean_Meta_foldImportedDecls___redArg___closed__1);
v___x_591_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_591_, 0, v___x_590_);
lean_ctor_set(v___x_591_, 1, v___x_589_);
lean_ctor_set(v___x_591_, 2, v___x_588_);
return v___x_591_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_foldImportedDecls___redArg(lean_object* v_init_599_, lean_object* v_cfg_600_, lean_object* v_act_601_, lean_object* v_constantsPerTask_602_, lean_object* v_a_603_, lean_object* v_a_604_){
_start:
{
lean_object* v___x_606_; lean_object* v___x_607_; lean_object* v___x_608_; uint8_t v___x_609_; lean_object* v___x_610_; lean_object* v___x_611_; lean_object* v___x_612_; lean_object* v___x_613_; uint8_t v___x_614_; lean_object* v___x_615_; lean_object* v___x_616_; lean_object* v___x_617_; lean_object* v_tasks_619_; lean_object* v_env_622_; lean_object* v___x_623_; lean_object* v_moduleData_624_; lean_object* v___x_625_; lean_object* v___f_626_; lean_object* v___x_627_; lean_object* v___x_3292__overap_628_; lean_object* v___x_629_; 
v___x_606_ = lean_st_ref_get(v_a_604_);
v___x_607_ = lean_box(1);
v___x_608_ = l_Lean_Meta_Config_toConfigWithKey(v_cfg_600_);
v___x_609_ = 0;
v___x_610_ = lean_unsigned_to_nat(0u);
v___x_611_ = lean_obj_once(&lp_mathlib_Lean_Meta_foldImportedDecls___redArg___closed__4, &lp_mathlib_Lean_Meta_foldImportedDecls___redArg___closed__4_once, _init_lp_mathlib_Lean_Meta_foldImportedDecls___redArg___closed__4);
v___x_612_ = ((lean_object*)(lp_mathlib_Lean_Meta_foldImportedDecls___redArg___closed__5));
v___x_613_ = lean_box(0);
v___x_614_ = 1;
v___x_615_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v___x_615_, 0, v___x_608_);
lean_ctor_set(v___x_615_, 1, v___x_607_);
lean_ctor_set(v___x_615_, 2, v___x_611_);
lean_ctor_set(v___x_615_, 3, v___x_612_);
lean_ctor_set(v___x_615_, 4, v___x_613_);
lean_ctor_set(v___x_615_, 5, v___x_610_);
lean_ctor_set(v___x_615_, 6, v___x_613_);
lean_ctor_set_uint8(v___x_615_, sizeof(void*)*7, v___x_609_);
lean_ctor_set_uint8(v___x_615_, sizeof(void*)*7 + 1, v___x_609_);
lean_ctor_set_uint8(v___x_615_, sizeof(void*)*7 + 2, v___x_609_);
lean_ctor_set_uint8(v___x_615_, sizeof(void*)*7 + 3, v___x_614_);
v___x_616_ = lean_box(0);
v___x_617_ = lean_st_mk_ref(v___x_616_);
v_env_622_ = lean_ctor_get(v___x_606_, 0);
lean_inc_ref_n(v_env_622_, 2);
lean_dec(v___x_606_);
v___x_623_ = l_Lean_Environment_header(v_env_622_);
v_moduleData_624_ = lean_ctor_get(v___x_623_, 6);
lean_inc_ref(v_moduleData_624_);
lean_dec_ref(v___x_623_);
v___x_625_ = lean_array_get_size(v_moduleData_624_);
lean_inc_ref_n(v_a_603_, 2);
lean_inc_ref(v___x_615_);
lean_inc_ref(v_act_601_);
lean_inc(v_init_599_);
lean_inc(v___x_617_);
v___f_626_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Meta_foldImportedDecls___redArg___lam__0___boxed), 17, 10);
lean_closure_set(v___f_626_, 0, v___x_625_);
lean_closure_set(v___f_626_, 1, v_moduleData_624_);
lean_closure_set(v___f_626_, 2, v_constantsPerTask_602_);
lean_closure_set(v___f_626_, 3, v___x_617_);
lean_closure_set(v___f_626_, 4, v_env_622_);
lean_closure_set(v___f_626_, 5, v_init_599_);
lean_closure_set(v___f_626_, 6, v_act_601_);
lean_closure_set(v___f_626_, 7, v___x_615_);
lean_closure_set(v___f_626_, 8, v_a_603_);
lean_closure_set(v___f_626_, 9, v___x_610_);
v___x_627_ = ((lean_object*)(lp_mathlib_Lean_Meta_foldImportedDecls___redArg___closed__7));
v___x_3292__overap_628_ = l_WellFounded_opaqueFix_u2083___redArg(v___f_626_, v___x_610_, v___x_627_, lean_box(0));
lean_inc(v_a_604_);
v___x_629_ = lean_apply_3(v___x_3292__overap_628_, v_a_603_, v_a_604_, lean_box(0));
if (lean_obj_tag(v___x_629_) == 0)
{
lean_object* v_a_630_; lean_object* v_snd_631_; lean_object* v_fst_632_; lean_object* v_fst_633_; lean_object* v___x_635_; uint8_t v_isShared_636_; uint8_t v_isSharedCheck_677_; 
v_a_630_ = lean_ctor_get(v___x_629_, 0);
lean_inc(v_a_630_);
lean_dec_ref_known(v___x_629_, 1);
v_snd_631_ = lean_ctor_get(v_a_630_, 1);
lean_inc(v_snd_631_);
v_fst_632_ = lean_ctor_get(v_a_630_, 0);
lean_inc(v_fst_632_);
lean_dec(v_a_630_);
v_fst_633_ = lean_ctor_get(v_snd_631_, 0);
v_isSharedCheck_677_ = !lean_is_exclusive(v_snd_631_);
if (v_isSharedCheck_677_ == 0)
{
lean_object* v_unused_678_; 
v_unused_678_ = lean_ctor_get(v_snd_631_, 1);
lean_dec(v_unused_678_);
v___x_635_ = v_snd_631_;
v_isShared_636_ = v_isSharedCheck_677_;
goto v_resetjp_634_;
}
else
{
lean_inc(v_fst_633_);
lean_dec(v_snd_631_);
v___x_635_ = lean_box(0);
v_isShared_636_ = v_isSharedCheck_677_;
goto v_resetjp_634_;
}
v_resetjp_634_:
{
uint8_t v___x_637_; 
v___x_637_ = lean_nat_dec_lt(v_fst_633_, v___x_625_);
if (v___x_637_ == 0)
{
lean_del_object(v___x_635_);
lean_dec(v_fst_633_);
lean_dec_ref(v_env_622_);
lean_dec_ref_known(v___x_615_, 7);
lean_dec_ref(v_act_601_);
lean_dec(v_init_599_);
v_tasks_619_ = v_fst_632_;
goto v___jp_618_;
}
else
{
lean_object* v___x_638_; lean_object* v___x_639_; lean_object* v_ngen_640_; lean_object* v_namePrefix_641_; lean_object* v_idx_642_; lean_object* v___x_644_; uint8_t v_isShared_645_; uint8_t v_isSharedCheck_676_; 
v___x_638_ = lean_st_ref_get(v_a_604_);
v___x_639_ = lean_st_ref_take(v_a_604_);
v_ngen_640_ = lean_ctor_get(v___x_638_, 2);
lean_inc_ref(v_ngen_640_);
lean_dec(v___x_638_);
v_namePrefix_641_ = lean_ctor_get(v_ngen_640_, 0);
v_idx_642_ = lean_ctor_get(v_ngen_640_, 1);
v_isSharedCheck_676_ = !lean_is_exclusive(v_ngen_640_);
if (v_isSharedCheck_676_ == 0)
{
v___x_644_ = v_ngen_640_;
v_isShared_645_ = v_isSharedCheck_676_;
goto v_resetjp_643_;
}
else
{
lean_inc(v_idx_642_);
lean_inc(v_namePrefix_641_);
lean_dec(v_ngen_640_);
v___x_644_ = lean_box(0);
v_isShared_645_ = v_isSharedCheck_676_;
goto v_resetjp_643_;
}
v_resetjp_643_:
{
lean_object* v_env_646_; lean_object* v_nextMacroScope_647_; lean_object* v_auxDeclNGen_648_; lean_object* v_traceState_649_; lean_object* v_cache_650_; lean_object* v_messages_651_; lean_object* v_infoState_652_; lean_object* v_snapshotTasks_653_; lean_object* v___x_655_; uint8_t v_isShared_656_; uint8_t v_isSharedCheck_674_; 
v_env_646_ = lean_ctor_get(v___x_639_, 0);
v_nextMacroScope_647_ = lean_ctor_get(v___x_639_, 1);
v_auxDeclNGen_648_ = lean_ctor_get(v___x_639_, 3);
v_traceState_649_ = lean_ctor_get(v___x_639_, 4);
v_cache_650_ = lean_ctor_get(v___x_639_, 5);
v_messages_651_ = lean_ctor_get(v___x_639_, 6);
v_infoState_652_ = lean_ctor_get(v___x_639_, 7);
v_snapshotTasks_653_ = lean_ctor_get(v___x_639_, 8);
v_isSharedCheck_674_ = !lean_is_exclusive(v___x_639_);
if (v_isSharedCheck_674_ == 0)
{
lean_object* v_unused_675_; 
v_unused_675_ = lean_ctor_get(v___x_639_, 2);
lean_dec(v_unused_675_);
v___x_655_ = v___x_639_;
v_isShared_656_ = v_isSharedCheck_674_;
goto v_resetjp_654_;
}
else
{
lean_inc(v_snapshotTasks_653_);
lean_inc(v_infoState_652_);
lean_inc(v_messages_651_);
lean_inc(v_cache_650_);
lean_inc(v_traceState_649_);
lean_inc(v_auxDeclNGen_648_);
lean_inc(v_nextMacroScope_647_);
lean_inc(v_env_646_);
lean_dec(v___x_639_);
v___x_655_ = lean_box(0);
v_isShared_656_ = v_isSharedCheck_674_;
goto v_resetjp_654_;
}
v_resetjp_654_:
{
lean_object* v___x_657_; lean_object* v___x_658_; lean_object* v___x_659_; lean_object* v___x_661_; 
lean_inc(v_idx_642_);
lean_inc(v_namePrefix_641_);
v___x_657_ = l_Lean_Name_num___override(v_namePrefix_641_, v_idx_642_);
v___x_658_ = lean_unsigned_to_nat(1u);
v___x_659_ = lean_nat_add(v_idx_642_, v___x_658_);
lean_dec(v_idx_642_);
if (v_isShared_645_ == 0)
{
lean_ctor_set(v___x_644_, 1, v___x_659_);
v___x_661_ = v___x_644_;
goto v_reusejp_660_;
}
else
{
lean_object* v_reuseFailAlloc_673_; 
v_reuseFailAlloc_673_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_673_, 0, v_namePrefix_641_);
lean_ctor_set(v_reuseFailAlloc_673_, 1, v___x_659_);
v___x_661_ = v_reuseFailAlloc_673_;
goto v_reusejp_660_;
}
v_reusejp_660_:
{
lean_object* v___x_663_; 
if (v_isShared_656_ == 0)
{
lean_ctor_set(v___x_655_, 2, v___x_661_);
v___x_663_ = v___x_655_;
goto v_reusejp_662_;
}
else
{
lean_object* v_reuseFailAlloc_672_; 
v_reuseFailAlloc_672_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_672_, 0, v_env_646_);
lean_ctor_set(v_reuseFailAlloc_672_, 1, v_nextMacroScope_647_);
lean_ctor_set(v_reuseFailAlloc_672_, 2, v___x_661_);
lean_ctor_set(v_reuseFailAlloc_672_, 3, v_auxDeclNGen_648_);
lean_ctor_set(v_reuseFailAlloc_672_, 4, v_traceState_649_);
lean_ctor_set(v_reuseFailAlloc_672_, 5, v_cache_650_);
lean_ctor_set(v_reuseFailAlloc_672_, 6, v_messages_651_);
lean_ctor_set(v_reuseFailAlloc_672_, 7, v_infoState_652_);
lean_ctor_set(v_reuseFailAlloc_672_, 8, v_snapshotTasks_653_);
v___x_663_ = v_reuseFailAlloc_672_;
goto v_reusejp_662_;
}
v_reusejp_662_:
{
lean_object* v___x_664_; lean_object* v___x_666_; 
v___x_664_ = lean_st_ref_set(v_a_604_, v___x_663_);
if (v_isShared_636_ == 0)
{
lean_ctor_set(v___x_635_, 1, v___x_658_);
lean_ctor_set(v___x_635_, 0, v___x_657_);
v___x_666_ = v___x_635_;
goto v_reusejp_665_;
}
else
{
lean_object* v_reuseFailAlloc_671_; 
v_reuseFailAlloc_671_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_671_, 0, v___x_657_);
lean_ctor_set(v_reuseFailAlloc_671_, 1, v___x_658_);
v___x_666_ = v_reuseFailAlloc_671_;
goto v_reusejp_665_;
}
v_reusejp_665_:
{
lean_object* v___x_667_; lean_object* v___x_668_; lean_object* v___x_669_; lean_object* v___x_670_; 
lean_inc_ref(v_a_603_);
lean_inc(v___x_617_);
v___x_667_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___boxed), 11, 10);
lean_closure_set(v___x_667_, 0, lean_box(0));
lean_closure_set(v___x_667_, 1, v___x_666_);
lean_closure_set(v___x_667_, 2, v___x_617_);
lean_closure_set(v___x_667_, 3, v_env_622_);
lean_closure_set(v___x_667_, 4, v_init_599_);
lean_closure_set(v___x_667_, 5, v_act_601_);
lean_closure_set(v___x_667_, 6, v___x_615_);
lean_closure_set(v___x_667_, 7, v_a_603_);
lean_closure_set(v___x_667_, 8, v_fst_633_);
lean_closure_set(v___x_667_, 9, v___x_625_);
v___x_668_ = lean_alloc_closure((void*)(l_EIO_toBaseIO___boxed), 4, 3);
lean_closure_set(v___x_668_, 0, lean_box(0));
lean_closure_set(v___x_668_, 1, lean_box(0));
lean_closure_set(v___x_668_, 2, v___x_667_);
v___x_669_ = lean_io_as_task(v___x_668_, v___x_610_);
v___x_670_ = lean_array_push(v_fst_632_, v___x_669_);
v_tasks_619_ = v___x_670_;
goto v___jp_618_;
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
lean_object* v_a_679_; lean_object* v___x_681_; uint8_t v_isShared_682_; uint8_t v_isSharedCheck_686_; 
lean_dec_ref(v_env_622_);
lean_dec(v___x_617_);
lean_dec_ref_known(v___x_615_, 7);
lean_dec_ref(v_act_601_);
lean_dec(v_init_599_);
v_a_679_ = lean_ctor_get(v___x_629_, 0);
v_isSharedCheck_686_ = !lean_is_exclusive(v___x_629_);
if (v_isSharedCheck_686_ == 0)
{
v___x_681_ = v___x_629_;
v_isShared_682_ = v_isSharedCheck_686_;
goto v_resetjp_680_;
}
else
{
lean_inc(v_a_679_);
lean_dec(v___x_629_);
v___x_681_ = lean_box(0);
v_isShared_682_ = v_isSharedCheck_686_;
goto v_resetjp_680_;
}
v_resetjp_680_:
{
lean_object* v___x_684_; 
if (v_isShared_682_ == 0)
{
v___x_684_ = v___x_681_;
goto v_reusejp_683_;
}
else
{
lean_object* v_reuseFailAlloc_685_; 
v_reuseFailAlloc_685_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_685_, 0, v_a_679_);
v___x_684_ = v_reuseFailAlloc_685_;
goto v_reusejp_683_;
}
v_reusejp_683_:
{
return v___x_684_;
}
}
}
v___jp_618_:
{
lean_object* v___x_620_; lean_object* v___x_621_; 
v___x_620_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_620_, 0, v_tasks_619_);
lean_ctor_set(v___x_620_, 1, v___x_617_);
v___x_621_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_621_, 0, v___x_620_);
return v___x_621_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_foldImportedDecls___redArg___boxed(lean_object* v_init_687_, lean_object* v_cfg_688_, lean_object* v_act_689_, lean_object* v_constantsPerTask_690_, lean_object* v_a_691_, lean_object* v_a_692_, lean_object* v_a_693_){
_start:
{
lean_object* v_res_694_; 
v_res_694_ = lp_mathlib_Lean_Meta_foldImportedDecls___redArg(v_init_687_, v_cfg_688_, v_act_689_, v_constantsPerTask_690_, v_a_691_, v_a_692_);
lean_dec(v_a_692_);
lean_dec_ref(v_a_691_);
return v_res_694_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_foldImportedDecls(lean_object* v_00_u03b1_695_, lean_object* v_init_696_, lean_object* v_cfg_697_, lean_object* v_act_698_, lean_object* v_constantsPerTask_699_, lean_object* v_a_700_, lean_object* v_a_701_){
_start:
{
lean_object* v___x_703_; 
v___x_703_ = lp_mathlib_Lean_Meta_foldImportedDecls___redArg(v_init_696_, v_cfg_697_, v_act_698_, v_constantsPerTask_699_, v_a_700_, v_a_701_);
return v___x_703_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_foldImportedDecls___boxed(lean_object* v_00_u03b1_704_, lean_object* v_init_705_, lean_object* v_cfg_706_, lean_object* v_act_707_, lean_object* v_constantsPerTask_708_, lean_object* v_a_709_, lean_object* v_a_710_, lean_object* v_a_711_){
_start:
{
lean_object* v_res_712_; 
v_res_712_ = lp_mathlib_Lean_Meta_foldImportedDecls(v_00_u03b1_704_, v_init_705_, v_cfg_706_, v_act_707_, v_constantsPerTask_708_, v_a_709_, v_a_710_);
lean_dec(v_a_710_);
lean_dec_ref(v_a_709_);
return v_res_712_;
}
}
static lean_object* _init_lp_mathlib_Lean_Meta_foldCurrFileDecls___redArg___closed__2(void){
_start:
{
lean_object* v___x_715_; lean_object* v___x_716_; lean_object* v___x_717_; 
v___x_715_ = lean_obj_once(&lp_mathlib_Lean_Meta_foldImportedDecls___redArg___closed__1, &lp_mathlib_Lean_Meta_foldImportedDecls___redArg___closed__1_once, _init_lp_mathlib_Lean_Meta_foldImportedDecls___redArg___closed__1);
v___x_716_ = lean_unsigned_to_nat(0u);
v___x_717_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v___x_717_, 0, v___x_716_);
lean_ctor_set(v___x_717_, 1, v___x_716_);
lean_ctor_set(v___x_717_, 2, v___x_716_);
lean_ctor_set(v___x_717_, 3, v___x_716_);
lean_ctor_set(v___x_717_, 4, v___x_715_);
lean_ctor_set(v___x_717_, 5, v___x_715_);
lean_ctor_set(v___x_717_, 6, v___x_715_);
lean_ctor_set(v___x_717_, 7, v___x_715_);
lean_ctor_set(v___x_717_, 8, v___x_715_);
lean_ctor_set(v___x_717_, 9, v___x_715_);
return v___x_717_;
}
}
static lean_object* _init_lp_mathlib_Lean_Meta_foldCurrFileDecls___redArg___closed__3(void){
_start:
{
lean_object* v___x_718_; lean_object* v___x_719_; 
v___x_718_ = lean_obj_once(&lp_mathlib_Lean_Meta_foldImportedDecls___redArg___closed__1, &lp_mathlib_Lean_Meta_foldImportedDecls___redArg___closed__1_once, _init_lp_mathlib_Lean_Meta_foldImportedDecls___redArg___closed__1);
v___x_719_ = lean_alloc_ctor(0, 6, 0);
lean_ctor_set(v___x_719_, 0, v___x_718_);
lean_ctor_set(v___x_719_, 1, v___x_718_);
lean_ctor_set(v___x_719_, 2, v___x_718_);
lean_ctor_set(v___x_719_, 3, v___x_718_);
lean_ctor_set(v___x_719_, 4, v___x_718_);
lean_ctor_set(v___x_719_, 5, v___x_718_);
return v___x_719_;
}
}
static lean_object* _init_lp_mathlib_Lean_Meta_foldCurrFileDecls___redArg___closed__4(void){
_start:
{
lean_object* v___x_720_; lean_object* v___x_721_; 
v___x_720_ = lean_obj_once(&lp_mathlib_Lean_Meta_foldImportedDecls___redArg___closed__1, &lp_mathlib_Lean_Meta_foldImportedDecls___redArg___closed__1_once, _init_lp_mathlib_Lean_Meta_foldImportedDecls___redArg___closed__1);
v___x_721_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_721_, 0, v___x_720_);
lean_ctor_set(v___x_721_, 1, v___x_720_);
lean_ctor_set(v___x_721_, 2, v___x_720_);
lean_ctor_set(v___x_721_, 3, v___x_720_);
lean_ctor_set(v___x_721_, 4, v___x_720_);
return v___x_721_;
}
}
static lean_object* _init_lp_mathlib_Lean_Meta_foldCurrFileDecls___redArg___closed__6(void){
_start:
{
lean_object* v___x_726_; uint64_t v___x_727_; lean_object* v___x_728_; 
v___x_726_ = lean_obj_once(&lp_mathlib_Lean_Meta_foldImportedDecls___redArg___closed__3, &lp_mathlib_Lean_Meta_foldImportedDecls___redArg___closed__3_once, _init_lp_mathlib_Lean_Meta_foldImportedDecls___redArg___closed__3);
v___x_727_ = 0ULL;
v___x_728_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v___x_728_, 0, v___x_726_);
lean_ctor_set_uint64(v___x_728_, sizeof(void*)*1, v___x_727_);
return v___x_728_;
}
}
static lean_object* _init_lp_mathlib_Lean_Meta_foldCurrFileDecls___redArg___closed__7(void){
_start:
{
lean_object* v___x_729_; lean_object* v___x_730_; 
v___x_729_ = lean_obj_once(&lp_mathlib_Lean_Meta_foldImportedDecls___redArg___closed__1, &lp_mathlib_Lean_Meta_foldImportedDecls___redArg___closed__1_once, _init_lp_mathlib_Lean_Meta_foldImportedDecls___redArg___closed__1);
v___x_730_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_730_, 0, v___x_729_);
lean_ctor_set(v___x_730_, 1, v___x_729_);
return v___x_730_;
}
}
static lean_object* _init_lp_mathlib_Lean_Meta_foldCurrFileDecls___redArg___closed__8(void){
_start:
{
lean_object* v___x_731_; lean_object* v___x_732_; lean_object* v___x_733_; 
v___x_731_ = l_Lean_NameSet_empty;
v___x_732_ = lean_obj_once(&lp_mathlib_Lean_Meta_foldImportedDecls___redArg___closed__3, &lp_mathlib_Lean_Meta_foldImportedDecls___redArg___closed__3_once, _init_lp_mathlib_Lean_Meta_foldImportedDecls___redArg___closed__3);
v___x_733_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_733_, 0, v___x_732_);
lean_ctor_set(v___x_733_, 1, v___x_732_);
lean_ctor_set(v___x_733_, 2, v___x_731_);
return v___x_733_;
}
}
static lean_object* _init_lp_mathlib_Lean_Meta_foldCurrFileDecls___redArg___closed__9(void){
_start:
{
lean_object* v___x_734_; lean_object* v___x_735_; uint8_t v___x_736_; lean_object* v___x_737_; 
v___x_734_ = lean_obj_once(&lp_mathlib_Lean_Meta_foldImportedDecls___redArg___closed__3, &lp_mathlib_Lean_Meta_foldImportedDecls___redArg___closed__3_once, _init_lp_mathlib_Lean_Meta_foldImportedDecls___redArg___closed__3);
v___x_735_ = lean_obj_once(&lp_mathlib_Lean_Meta_foldImportedDecls___redArg___closed__1, &lp_mathlib_Lean_Meta_foldImportedDecls___redArg___closed__1_once, _init_lp_mathlib_Lean_Meta_foldImportedDecls___redArg___closed__1);
v___x_736_ = 1;
v___x_737_ = lean_alloc_ctor(0, 3, 1);
lean_ctor_set(v___x_737_, 0, v___x_735_);
lean_ctor_set(v___x_737_, 1, v___x_735_);
lean_ctor_set(v___x_737_, 2, v___x_734_);
lean_ctor_set_uint8(v___x_737_, sizeof(void*)*3, v___x_736_);
return v___x_737_;
}
}
static lean_object* _init_lp_mathlib_Lean_Meta_foldCurrFileDecls___redArg___closed__10(void){
_start:
{
lean_object* v___x_738_; lean_object* v___x_739_; lean_object* v___x_740_; lean_object* v___x_741_; lean_object* v___x_742_; lean_object* v___x_743_; 
v___x_738_ = lean_obj_once(&lp_mathlib_Lean_Meta_foldCurrFileDecls___redArg___closed__4, &lp_mathlib_Lean_Meta_foldCurrFileDecls___redArg___closed__4_once, _init_lp_mathlib_Lean_Meta_foldCurrFileDecls___redArg___closed__4);
v___x_739_ = lean_obj_once(&lp_mathlib_Lean_Meta_foldImportedDecls___redArg___closed__3, &lp_mathlib_Lean_Meta_foldImportedDecls___redArg___closed__3_once, _init_lp_mathlib_Lean_Meta_foldImportedDecls___redArg___closed__3);
v___x_740_ = lean_box(1);
v___x_741_ = lean_obj_once(&lp_mathlib_Lean_Meta_foldCurrFileDecls___redArg___closed__3, &lp_mathlib_Lean_Meta_foldCurrFileDecls___redArg___closed__3_once, _init_lp_mathlib_Lean_Meta_foldCurrFileDecls___redArg___closed__3);
v___x_742_ = lean_obj_once(&lp_mathlib_Lean_Meta_foldCurrFileDecls___redArg___closed__2, &lp_mathlib_Lean_Meta_foldCurrFileDecls___redArg___closed__2_once, _init_lp_mathlib_Lean_Meta_foldCurrFileDecls___redArg___closed__2);
v___x_743_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_743_, 0, v___x_742_);
lean_ctor_set(v___x_743_, 1, v___x_741_);
lean_ctor_set(v___x_743_, 2, v___x_740_);
lean_ctor_set(v___x_743_, 3, v___x_739_);
lean_ctor_set(v___x_743_, 4, v___x_738_);
return v___x_743_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_foldCurrFileDecls___redArg(lean_object* v_init_744_, lean_object* v_cfg_745_, lean_object* v_act_746_, lean_object* v_a_747_, lean_object* v_a_748_){
_start:
{
lean_object* v___x_750_; lean_object* v_toApplicative_751_; lean_object* v_toFunctor_752_; lean_object* v_toSeq_753_; lean_object* v_toSeqLeft_754_; lean_object* v_toSeqRight_755_; lean_object* v___f_756_; lean_object* v___f_757_; lean_object* v___f_758_; lean_object* v___f_759_; lean_object* v___x_760_; lean_object* v___f_761_; lean_object* v___f_762_; lean_object* v___f_763_; lean_object* v___x_764_; lean_object* v___x_765_; lean_object* v___x_766_; lean_object* v_toApplicative_767_; lean_object* v___x_769_; uint8_t v_isShared_770_; uint8_t v_isSharedCheck_939_; 
v___x_750_ = lean_obj_once(&lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___lam__1___closed__1, &lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___lam__1___closed__1_once, _init_lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___lam__1___closed__1);
v_toApplicative_751_ = lean_ctor_get(v___x_750_, 0);
v_toFunctor_752_ = lean_ctor_get(v_toApplicative_751_, 0);
v_toSeq_753_ = lean_ctor_get(v_toApplicative_751_, 2);
v_toSeqLeft_754_ = lean_ctor_get(v_toApplicative_751_, 3);
v_toSeqRight_755_ = lean_ctor_get(v_toApplicative_751_, 4);
v___f_756_ = ((lean_object*)(lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___lam__1___closed__2));
v___f_757_ = ((lean_object*)(lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___lam__1___closed__3));
lean_inc_ref_n(v_toFunctor_752_, 2);
v___f_758_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(v___f_758_, 0, v_toFunctor_752_);
v___f_759_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_759_, 0, v_toFunctor_752_);
v___x_760_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_760_, 0, v___f_758_);
lean_ctor_set(v___x_760_, 1, v___f_759_);
lean_inc(v_toSeqRight_755_);
v___f_761_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_761_, 0, v_toSeqRight_755_);
lean_inc(v_toSeqLeft_754_);
v___f_762_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(v___f_762_, 0, v_toSeqLeft_754_);
lean_inc(v_toSeq_753_);
v___f_763_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(v___f_763_, 0, v_toSeq_753_);
v___x_764_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_764_, 0, v___x_760_);
lean_ctor_set(v___x_764_, 1, v___f_756_);
lean_ctor_set(v___x_764_, 2, v___f_763_);
lean_ctor_set(v___x_764_, 3, v___f_762_);
lean_ctor_set(v___x_764_, 4, v___f_761_);
v___x_765_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_765_, 0, v___x_764_);
lean_ctor_set(v___x_765_, 1, v___f_757_);
v___x_766_ = l_StateRefT_x27_instMonad___redArg(v___x_765_);
v_toApplicative_767_ = lean_ctor_get(v___x_766_, 0);
v_isSharedCheck_939_ = !lean_is_exclusive(v___x_766_);
if (v_isSharedCheck_939_ == 0)
{
lean_object* v_unused_940_; 
v_unused_940_ = lean_ctor_get(v___x_766_, 1);
lean_dec(v_unused_940_);
v___x_769_ = v___x_766_;
v_isShared_770_ = v_isSharedCheck_939_;
goto v_resetjp_768_;
}
else
{
lean_inc(v_toApplicative_767_);
lean_dec(v___x_766_);
v___x_769_ = lean_box(0);
v_isShared_770_ = v_isSharedCheck_939_;
goto v_resetjp_768_;
}
v_resetjp_768_:
{
lean_object* v_toFunctor_771_; lean_object* v_toSeq_772_; lean_object* v_toSeqLeft_773_; lean_object* v_toSeqRight_774_; lean_object* v___x_776_; uint8_t v_isShared_777_; uint8_t v_isSharedCheck_937_; 
v_toFunctor_771_ = lean_ctor_get(v_toApplicative_767_, 0);
v_toSeq_772_ = lean_ctor_get(v_toApplicative_767_, 2);
v_toSeqLeft_773_ = lean_ctor_get(v_toApplicative_767_, 3);
v_toSeqRight_774_ = lean_ctor_get(v_toApplicative_767_, 4);
v_isSharedCheck_937_ = !lean_is_exclusive(v_toApplicative_767_);
if (v_isSharedCheck_937_ == 0)
{
lean_object* v_unused_938_; 
v_unused_938_ = lean_ctor_get(v_toApplicative_767_, 1);
lean_dec(v_unused_938_);
v___x_776_ = v_toApplicative_767_;
v_isShared_777_ = v_isSharedCheck_937_;
goto v_resetjp_775_;
}
else
{
lean_inc(v_toSeqRight_774_);
lean_inc(v_toSeqLeft_773_);
lean_inc(v_toSeq_772_);
lean_inc(v_toFunctor_771_);
lean_dec(v_toApplicative_767_);
v___x_776_ = lean_box(0);
v_isShared_777_ = v_isSharedCheck_937_;
goto v_resetjp_775_;
}
v_resetjp_775_:
{
lean_object* v___f_778_; lean_object* v___f_779_; lean_object* v___f_780_; lean_object* v___f_781_; lean_object* v___x_782_; lean_object* v___f_783_; lean_object* v___f_784_; lean_object* v___f_785_; lean_object* v___x_787_; 
v___f_778_ = ((lean_object*)(lp_mathlib_Lean_Meta_foldCurrFileDecls___redArg___closed__0));
v___f_779_ = ((lean_object*)(lp_mathlib_Lean_Meta_foldCurrFileDecls___redArg___closed__1));
lean_inc_ref(v_toFunctor_771_);
v___f_780_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(v___f_780_, 0, v_toFunctor_771_);
v___f_781_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_781_, 0, v_toFunctor_771_);
v___x_782_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_782_, 0, v___f_780_);
lean_ctor_set(v___x_782_, 1, v___f_781_);
v___f_783_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_783_, 0, v_toSeqRight_774_);
v___f_784_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(v___f_784_, 0, v_toSeqLeft_773_);
v___f_785_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(v___f_785_, 0, v_toSeq_772_);
if (v_isShared_777_ == 0)
{
lean_ctor_set(v___x_776_, 4, v___f_783_);
lean_ctor_set(v___x_776_, 3, v___f_784_);
lean_ctor_set(v___x_776_, 2, v___f_785_);
lean_ctor_set(v___x_776_, 1, v___f_778_);
lean_ctor_set(v___x_776_, 0, v___x_782_);
v___x_787_ = v___x_776_;
goto v_reusejp_786_;
}
else
{
lean_object* v_reuseFailAlloc_936_; 
v_reuseFailAlloc_936_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_936_, 0, v___x_782_);
lean_ctor_set(v_reuseFailAlloc_936_, 1, v___f_778_);
lean_ctor_set(v_reuseFailAlloc_936_, 2, v___f_785_);
lean_ctor_set(v_reuseFailAlloc_936_, 3, v___f_784_);
lean_ctor_set(v_reuseFailAlloc_936_, 4, v___f_783_);
v___x_787_ = v_reuseFailAlloc_936_;
goto v_reusejp_786_;
}
v_reusejp_786_:
{
lean_object* v___x_789_; 
if (v_isShared_770_ == 0)
{
lean_ctor_set(v___x_769_, 1, v___f_779_);
lean_ctor_set(v___x_769_, 0, v___x_787_);
v___x_789_ = v___x_769_;
goto v_reusejp_788_;
}
else
{
lean_object* v_reuseFailAlloc_935_; 
v_reuseFailAlloc_935_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_935_, 0, v___x_787_);
lean_ctor_set(v_reuseFailAlloc_935_, 1, v___f_779_);
v___x_789_ = v_reuseFailAlloc_935_;
goto v_reusejp_788_;
}
v_reusejp_788_:
{
lean_object* v___x_790_; lean_object* v___x_791_; lean_object* v___x_792_; lean_object* v___x_793_; lean_object* v_ngen_794_; lean_object* v_namePrefix_795_; lean_object* v_idx_796_; lean_object* v___x_798_; uint8_t v_isShared_799_; uint8_t v_isSharedCheck_934_; 
v___x_790_ = lean_st_ref_get(v_a_748_);
v___x_791_ = lean_box(0);
v___x_792_ = lean_st_mk_ref(v___x_791_);
v___x_793_ = lean_st_ref_get(v_a_748_);
v_ngen_794_ = lean_ctor_get(v___x_793_, 2);
lean_inc_ref(v_ngen_794_);
lean_dec(v___x_793_);
v_namePrefix_795_ = lean_ctor_get(v_ngen_794_, 0);
v_idx_796_ = lean_ctor_get(v_ngen_794_, 1);
v_isSharedCheck_934_ = !lean_is_exclusive(v_ngen_794_);
if (v_isSharedCheck_934_ == 0)
{
v___x_798_ = v_ngen_794_;
v_isShared_799_ = v_isSharedCheck_934_;
goto v_resetjp_797_;
}
else
{
lean_inc(v_idx_796_);
lean_inc(v_namePrefix_795_);
lean_dec(v_ngen_794_);
v___x_798_ = lean_box(0);
v_isShared_799_ = v_isSharedCheck_934_;
goto v_resetjp_797_;
}
v_resetjp_797_:
{
lean_object* v___x_800_; lean_object* v_env_801_; lean_object* v_nextMacroScope_802_; lean_object* v_auxDeclNGen_803_; lean_object* v_traceState_804_; lean_object* v_cache_805_; lean_object* v_messages_806_; lean_object* v_infoState_807_; lean_object* v_snapshotTasks_808_; lean_object* v___x_810_; uint8_t v_isShared_811_; uint8_t v_isSharedCheck_932_; 
v___x_800_ = lean_st_ref_take(v_a_748_);
v_env_801_ = lean_ctor_get(v___x_800_, 0);
v_nextMacroScope_802_ = lean_ctor_get(v___x_800_, 1);
v_auxDeclNGen_803_ = lean_ctor_get(v___x_800_, 3);
v_traceState_804_ = lean_ctor_get(v___x_800_, 4);
v_cache_805_ = lean_ctor_get(v___x_800_, 5);
v_messages_806_ = lean_ctor_get(v___x_800_, 6);
v_infoState_807_ = lean_ctor_get(v___x_800_, 7);
v_snapshotTasks_808_ = lean_ctor_get(v___x_800_, 8);
v_isSharedCheck_932_ = !lean_is_exclusive(v___x_800_);
if (v_isSharedCheck_932_ == 0)
{
lean_object* v_unused_933_; 
v_unused_933_ = lean_ctor_get(v___x_800_, 2);
lean_dec(v_unused_933_);
v___x_810_ = v___x_800_;
v_isShared_811_ = v_isSharedCheck_932_;
goto v_resetjp_809_;
}
else
{
lean_inc(v_snapshotTasks_808_);
lean_inc(v_infoState_807_);
lean_inc(v_messages_806_);
lean_inc(v_cache_805_);
lean_inc(v_traceState_804_);
lean_inc(v_auxDeclNGen_803_);
lean_inc(v_nextMacroScope_802_);
lean_inc(v_env_801_);
lean_dec(v___x_800_);
v___x_810_ = lean_box(0);
v_isShared_811_ = v_isSharedCheck_932_;
goto v_resetjp_809_;
}
v_resetjp_809_:
{
lean_object* v___x_812_; lean_object* v___x_813_; lean_object* v___x_814_; lean_object* v___x_816_; 
lean_inc(v_idx_796_);
lean_inc(v_namePrefix_795_);
v___x_812_ = l_Lean_Name_num___override(v_namePrefix_795_, v_idx_796_);
v___x_813_ = lean_unsigned_to_nat(1u);
v___x_814_ = lean_nat_add(v_idx_796_, v___x_813_);
lean_dec(v_idx_796_);
if (v_isShared_799_ == 0)
{
lean_ctor_set(v___x_798_, 1, v___x_814_);
v___x_816_ = v___x_798_;
goto v_reusejp_815_;
}
else
{
lean_object* v_reuseFailAlloc_931_; 
v_reuseFailAlloc_931_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_931_, 0, v_namePrefix_795_);
lean_ctor_set(v_reuseFailAlloc_931_, 1, v___x_814_);
v___x_816_ = v_reuseFailAlloc_931_;
goto v_reusejp_815_;
}
v_reusejp_815_:
{
lean_object* v___x_818_; 
if (v_isShared_811_ == 0)
{
lean_ctor_set(v___x_810_, 2, v___x_816_);
v___x_818_ = v___x_810_;
goto v_reusejp_817_;
}
else
{
lean_object* v_reuseFailAlloc_930_; 
v_reuseFailAlloc_930_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_930_, 0, v_env_801_);
lean_ctor_set(v_reuseFailAlloc_930_, 1, v_nextMacroScope_802_);
lean_ctor_set(v_reuseFailAlloc_930_, 2, v___x_816_);
lean_ctor_set(v_reuseFailAlloc_930_, 3, v_auxDeclNGen_803_);
lean_ctor_set(v_reuseFailAlloc_930_, 4, v_traceState_804_);
lean_ctor_set(v_reuseFailAlloc_930_, 5, v_cache_805_);
lean_ctor_set(v_reuseFailAlloc_930_, 6, v_messages_806_);
lean_ctor_set(v_reuseFailAlloc_930_, 7, v_infoState_807_);
lean_ctor_set(v_reuseFailAlloc_930_, 8, v_snapshotTasks_808_);
v___x_818_ = v_reuseFailAlloc_930_;
goto v_reusejp_817_;
}
v_reusejp_817_:
{
lean_object* v___x_819_; lean_object* v_env_820_; lean_object* v___x_822_; uint8_t v_isShared_823_; uint8_t v_isSharedCheck_921_; 
v___x_819_ = lean_st_ref_set(v_a_748_, v___x_818_);
v_env_820_ = lean_ctor_get(v___x_790_, 0);
v_isSharedCheck_921_ = !lean_is_exclusive(v___x_790_);
if (v_isSharedCheck_921_ == 0)
{
lean_object* v_unused_922_; lean_object* v_unused_923_; lean_object* v_unused_924_; lean_object* v_unused_925_; lean_object* v_unused_926_; lean_object* v_unused_927_; lean_object* v_unused_928_; lean_object* v_unused_929_; 
v_unused_922_ = lean_ctor_get(v___x_790_, 8);
lean_dec(v_unused_922_);
v_unused_923_ = lean_ctor_get(v___x_790_, 7);
lean_dec(v_unused_923_);
v_unused_924_ = lean_ctor_get(v___x_790_, 6);
lean_dec(v_unused_924_);
v_unused_925_ = lean_ctor_get(v___x_790_, 5);
lean_dec(v_unused_925_);
v_unused_926_ = lean_ctor_get(v___x_790_, 4);
lean_dec(v_unused_926_);
v_unused_927_ = lean_ctor_get(v___x_790_, 3);
lean_dec(v_unused_927_);
v_unused_928_ = lean_ctor_get(v___x_790_, 2);
lean_dec(v_unused_928_);
v_unused_929_ = lean_ctor_get(v___x_790_, 1);
lean_dec(v_unused_929_);
v___x_822_ = v___x_790_;
v_isShared_823_ = v_isSharedCheck_921_;
goto v_resetjp_821_;
}
else
{
lean_inc(v_env_820_);
lean_dec(v___x_790_);
v___x_822_ = lean_box(0);
v_isShared_823_ = v_isSharedCheck_921_;
goto v_resetjp_821_;
}
v_resetjp_821_:
{
lean_object* v___x_824_; lean_object* v___x_825_; lean_object* v___x_826_; lean_object* v___x_827_; lean_object* v___x_828_; uint8_t v___x_829_; lean_object* v___x_830_; lean_object* v___x_831_; lean_object* v___x_832_; lean_object* v___x_833_; uint8_t v___x_834_; lean_object* v___x_835_; lean_object* v___x_836_; lean_object* v___x_837_; lean_object* v___x_838_; lean_object* v___x_839_; lean_object* v___x_840_; lean_object* v___x_841_; lean_object* v___x_843_; 
v___x_824_ = lean_box(1);
v___x_825_ = l_Lean_Environment_header(v_env_820_);
v___x_826_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_826_, 0, v___x_812_);
lean_ctor_set(v___x_826_, 1, v___x_813_);
lean_inc_ref(v_env_820_);
v___x_827_ = l_Lean_Environment_constants(v_env_820_);
v___x_828_ = l_Lean_Meta_Config_toConfigWithKey(v_cfg_745_);
v___x_829_ = 0;
v___x_830_ = lean_unsigned_to_nat(0u);
v___x_831_ = lean_obj_once(&lp_mathlib_Lean_Meta_foldImportedDecls___redArg___closed__4, &lp_mathlib_Lean_Meta_foldImportedDecls___redArg___closed__4_once, _init_lp_mathlib_Lean_Meta_foldImportedDecls___redArg___closed__4);
v___x_832_ = ((lean_object*)(lp_mathlib_Lean_Meta_foldImportedDecls___redArg___closed__5));
v___x_833_ = lean_box(0);
v___x_834_ = 1;
v___x_835_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v___x_835_, 0, v___x_828_);
lean_ctor_set(v___x_835_, 1, v___x_824_);
lean_ctor_set(v___x_835_, 2, v___x_831_);
lean_ctor_set(v___x_835_, 3, v___x_832_);
lean_ctor_set(v___x_835_, 4, v___x_833_);
lean_ctor_set(v___x_835_, 5, v___x_830_);
lean_ctor_set(v___x_835_, 6, v___x_833_);
lean_ctor_set_uint8(v___x_835_, sizeof(void*)*7, v___x_829_);
lean_ctor_set_uint8(v___x_835_, sizeof(void*)*7 + 1, v___x_829_);
lean_ctor_set_uint8(v___x_835_, sizeof(void*)*7 + 2, v___x_829_);
lean_ctor_set_uint8(v___x_835_, sizeof(void*)*7 + 3, v___x_834_);
v___x_836_ = lean_obj_once(&lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___closed__7, &lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___closed__7_once, _init_lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_foldModules___redArg___closed__7);
v___x_837_ = ((lean_object*)(lp_mathlib_Lean_Meta_foldCurrFileDecls___redArg___closed__5));
v___x_838_ = lean_obj_once(&lp_mathlib_Lean_Meta_foldCurrFileDecls___redArg___closed__6, &lp_mathlib_Lean_Meta_foldCurrFileDecls___redArg___closed__6_once, _init_lp_mathlib_Lean_Meta_foldCurrFileDecls___redArg___closed__6);
v___x_839_ = lean_obj_once(&lp_mathlib_Lean_Meta_foldCurrFileDecls___redArg___closed__7, &lp_mathlib_Lean_Meta_foldCurrFileDecls___redArg___closed__7_once, _init_lp_mathlib_Lean_Meta_foldCurrFileDecls___redArg___closed__7);
v___x_840_ = lean_obj_once(&lp_mathlib_Lean_Meta_foldCurrFileDecls___redArg___closed__8, &lp_mathlib_Lean_Meta_foldCurrFileDecls___redArg___closed__8_once, _init_lp_mathlib_Lean_Meta_foldCurrFileDecls___redArg___closed__8);
v___x_841_ = lean_obj_once(&lp_mathlib_Lean_Meta_foldCurrFileDecls___redArg___closed__9, &lp_mathlib_Lean_Meta_foldCurrFileDecls___redArg___closed__9_once, _init_lp_mathlib_Lean_Meta_foldCurrFileDecls___redArg___closed__9);
if (v_isShared_823_ == 0)
{
lean_ctor_set(v___x_822_, 8, v___x_832_);
lean_ctor_set(v___x_822_, 7, v___x_841_);
lean_ctor_set(v___x_822_, 6, v___x_840_);
lean_ctor_set(v___x_822_, 5, v___x_839_);
lean_ctor_set(v___x_822_, 4, v___x_838_);
lean_ctor_set(v___x_822_, 3, v___x_837_);
lean_ctor_set(v___x_822_, 2, v___x_826_);
lean_ctor_set(v___x_822_, 1, v___x_836_);
v___x_843_ = v___x_822_;
goto v_reusejp_842_;
}
else
{
lean_object* v_reuseFailAlloc_920_; 
v_reuseFailAlloc_920_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_920_, 0, v_env_820_);
lean_ctor_set(v_reuseFailAlloc_920_, 1, v___x_836_);
lean_ctor_set(v_reuseFailAlloc_920_, 2, v___x_826_);
lean_ctor_set(v_reuseFailAlloc_920_, 3, v___x_837_);
lean_ctor_set(v_reuseFailAlloc_920_, 4, v___x_838_);
lean_ctor_set(v_reuseFailAlloc_920_, 5, v___x_839_);
lean_ctor_set(v_reuseFailAlloc_920_, 6, v___x_840_);
lean_ctor_set(v_reuseFailAlloc_920_, 7, v___x_841_);
lean_ctor_set(v_reuseFailAlloc_920_, 8, v___x_832_);
v___x_843_ = v_reuseFailAlloc_920_;
goto v_reusejp_842_;
}
v_reusejp_842_:
{
lean_object* v___x_844_; lean_object* v_a_846_; lean_object* v___x_850_; lean_object* v___x_851_; lean_object* v___x_852_; lean_object* v___x_853_; lean_object* v___x_854_; lean_object* v_mainModule_855_; lean_object* v_map_u2082_856_; lean_object* v_fileName_857_; lean_object* v_fileMap_858_; lean_object* v_options_859_; lean_object* v_currRecDepth_860_; lean_object* v_ref_861_; lean_object* v_currNamespace_862_; lean_object* v_openDecls_863_; lean_object* v_initHeartbeats_864_; lean_object* v_maxHeartbeats_865_; lean_object* v_quotContext_866_; lean_object* v_currMacroScope_867_; lean_object* v_cancelTk_x3f_868_; uint8_t v_suppressElabErrors_869_; lean_object* v_env_870_; lean_object* v___x_871_; lean_object* v___x_872_; lean_object* v___x_873_; lean_object* v___x_874_; lean_object* v___y_876_; uint8_t v___y_896_; uint8_t v___x_917_; 
v___x_844_ = lean_st_mk_ref(v___x_843_);
v___x_850_ = l_Lean_inheritedTraceOptions;
v___x_851_ = lean_st_ref_get(v___x_850_);
v___x_852_ = l_Lean_KVMap_instValueBool;
v___x_853_ = l_Lean_KVMap_instValueNat;
v___x_854_ = lean_st_ref_get(v___x_844_);
v_mainModule_855_ = lean_ctor_get(v___x_825_, 0);
lean_inc(v_mainModule_855_);
lean_dec_ref(v___x_825_);
v_map_u2082_856_ = lean_ctor_get(v___x_827_, 1);
lean_inc_ref(v_map_u2082_856_);
lean_dec_ref(v___x_827_);
v_fileName_857_ = lean_ctor_get(v_a_747_, 0);
v_fileMap_858_ = lean_ctor_get(v_a_747_, 1);
v_options_859_ = lean_ctor_get(v_a_747_, 2);
v_currRecDepth_860_ = lean_ctor_get(v_a_747_, 3);
v_ref_861_ = lean_ctor_get(v_a_747_, 5);
v_currNamespace_862_ = lean_ctor_get(v_a_747_, 6);
v_openDecls_863_ = lean_ctor_get(v_a_747_, 7);
v_initHeartbeats_864_ = lean_ctor_get(v_a_747_, 8);
v_maxHeartbeats_865_ = lean_ctor_get(v_a_747_, 9);
v_quotContext_866_ = lean_ctor_get(v_a_747_, 10);
v_currMacroScope_867_ = lean_ctor_get(v_a_747_, 11);
v_cancelTk_x3f_868_ = lean_ctor_get(v_a_747_, 12);
v_suppressElabErrors_869_ = lean_ctor_get_uint8(v_a_747_, sizeof(void*)*14 + 1);
v_env_870_ = lean_ctor_get(v___x_854_, 0);
lean_inc_ref(v_env_870_);
lean_dec(v___x_854_);
lean_inc(v___x_792_);
v___x_871_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Lean_FoldEnvironment_0__Lean_Meta_visitConst___boxed), 12, 4);
lean_closure_set(v___x_871_, 0, lean_box(0));
lean_closure_set(v___x_871_, 1, v_mainModule_855_);
lean_closure_set(v___x_871_, 2, v___x_792_);
lean_closure_set(v___x_871_, 3, v_act_746_);
v___x_872_ = lean_obj_once(&lp_mathlib_Lean_Meta_foldCurrFileDecls___redArg___closed__10, &lp_mathlib_Lean_Meta_foldCurrFileDecls___redArg___closed__10_once, _init_lp_mathlib_Lean_Meta_foldCurrFileDecls___redArg___closed__10);
v___x_873_ = l_Lean_diagnostics;
v___x_874_ = l_Lean_Option_get___redArg(v___x_852_, v_options_859_, v___x_873_);
v___x_917_ = l_Lean_Kernel_isDiagnosticsEnabled(v_env_870_);
lean_dec_ref(v_env_870_);
if (v___x_917_ == 0)
{
uint8_t v___x_918_; 
v___x_918_ = lean_unbox(v___x_874_);
if (v___x_918_ == 0)
{
lean_inc(v___x_844_);
v___y_876_ = v___x_844_;
goto v___jp_875_;
}
else
{
v___y_896_ = v___x_917_;
goto v___jp_895_;
}
}
else
{
uint8_t v___x_919_; 
v___x_919_ = lean_unbox(v___x_874_);
v___y_896_ = v___x_919_;
goto v___jp_895_;
}
v___jp_845_:
{
lean_object* v___x_847_; lean_object* v___x_848_; lean_object* v___x_849_; 
v___x_847_ = lean_st_ref_get(v___x_844_);
lean_dec(v___x_844_);
lean_dec(v___x_847_);
v___x_848_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_848_, 0, v_a_846_);
lean_ctor_set(v___x_848_, 1, v___x_792_);
v___x_849_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_849_, 0, v___x_848_);
return v___x_849_;
}
v___jp_875_:
{
lean_object* v___x_877_; lean_object* v___x_878_; lean_object* v___x_879_; lean_object* v___x_880_; uint8_t v___x_881_; lean_object* v___x_3729__overap_882_; lean_object* v___x_883_; 
v___x_877_ = lean_st_mk_ref(v___x_872_);
v___x_878_ = l_Lean_maxRecDepth;
v___x_879_ = l_Lean_Option_get___redArg(v___x_853_, v_options_859_, v___x_878_);
lean_inc(v_cancelTk_x3f_868_);
lean_inc(v_currMacroScope_867_);
lean_inc(v_quotContext_866_);
lean_inc(v_maxHeartbeats_865_);
lean_inc(v_initHeartbeats_864_);
lean_inc(v_openDecls_863_);
lean_inc(v_currNamespace_862_);
lean_inc(v_ref_861_);
lean_inc(v_currRecDepth_860_);
lean_inc_ref(v_options_859_);
lean_inc_ref(v_fileMap_858_);
lean_inc_ref(v_fileName_857_);
v___x_880_ = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(v___x_880_, 0, v_fileName_857_);
lean_ctor_set(v___x_880_, 1, v_fileMap_858_);
lean_ctor_set(v___x_880_, 2, v_options_859_);
lean_ctor_set(v___x_880_, 3, v_currRecDepth_860_);
lean_ctor_set(v___x_880_, 4, v___x_879_);
lean_ctor_set(v___x_880_, 5, v_ref_861_);
lean_ctor_set(v___x_880_, 6, v_currNamespace_862_);
lean_ctor_set(v___x_880_, 7, v_openDecls_863_);
lean_ctor_set(v___x_880_, 8, v_initHeartbeats_864_);
lean_ctor_set(v___x_880_, 9, v_maxHeartbeats_865_);
lean_ctor_set(v___x_880_, 10, v_quotContext_866_);
lean_ctor_set(v___x_880_, 11, v_currMacroScope_867_);
lean_ctor_set(v___x_880_, 12, v_cancelTk_x3f_868_);
lean_ctor_set(v___x_880_, 13, v___x_851_);
v___x_881_ = lean_unbox(v___x_874_);
lean_dec(v___x_874_);
lean_ctor_set_uint8(v___x_880_, sizeof(void*)*14, v___x_881_);
lean_ctor_set_uint8(v___x_880_, sizeof(void*)*14 + 1, v_suppressElabErrors_869_);
v___x_3729__overap_882_ = l_Lean_PersistentHashMap_foldlMAux___redArg(v___x_789_, v___x_871_, v_map_u2082_856_, v_init_744_);
lean_inc(v___x_877_);
v___x_883_ = lean_apply_5(v___x_3729__overap_882_, v___x_835_, v___x_877_, v___x_880_, v___y_876_, lean_box(0));
if (lean_obj_tag(v___x_883_) == 0)
{
lean_object* v_a_884_; lean_object* v___x_885_; 
v_a_884_ = lean_ctor_get(v___x_883_, 0);
lean_inc(v_a_884_);
lean_dec_ref_known(v___x_883_, 1);
v___x_885_ = lean_st_ref_get(v___x_877_);
lean_dec(v___x_877_);
lean_dec(v___x_885_);
v_a_846_ = v_a_884_;
goto v___jp_845_;
}
else
{
lean_dec(v___x_877_);
if (lean_obj_tag(v___x_883_) == 0)
{
lean_object* v_a_886_; 
v_a_886_ = lean_ctor_get(v___x_883_, 0);
lean_inc(v_a_886_);
lean_dec_ref_known(v___x_883_, 1);
v_a_846_ = v_a_886_;
goto v___jp_845_;
}
else
{
lean_object* v_a_887_; lean_object* v___x_889_; uint8_t v_isShared_890_; uint8_t v_isSharedCheck_894_; 
lean_dec(v___x_844_);
lean_dec(v___x_792_);
v_a_887_ = lean_ctor_get(v___x_883_, 0);
v_isSharedCheck_894_ = !lean_is_exclusive(v___x_883_);
if (v_isSharedCheck_894_ == 0)
{
v___x_889_ = v___x_883_;
v_isShared_890_ = v_isSharedCheck_894_;
goto v_resetjp_888_;
}
else
{
lean_inc(v_a_887_);
lean_dec(v___x_883_);
v___x_889_ = lean_box(0);
v_isShared_890_ = v_isSharedCheck_894_;
goto v_resetjp_888_;
}
v_resetjp_888_:
{
lean_object* v___x_892_; 
if (v_isShared_890_ == 0)
{
v___x_892_ = v___x_889_;
goto v_reusejp_891_;
}
else
{
lean_object* v_reuseFailAlloc_893_; 
v_reuseFailAlloc_893_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_893_, 0, v_a_887_);
v___x_892_ = v_reuseFailAlloc_893_;
goto v_reusejp_891_;
}
v_reusejp_891_:
{
return v___x_892_;
}
}
}
}
}
v___jp_895_:
{
if (v___y_896_ == 0)
{
lean_object* v___x_897_; lean_object* v_env_898_; lean_object* v_nextMacroScope_899_; lean_object* v_ngen_900_; lean_object* v_auxDeclNGen_901_; lean_object* v_traceState_902_; lean_object* v_messages_903_; lean_object* v_infoState_904_; lean_object* v_snapshotTasks_905_; lean_object* v___x_907_; uint8_t v_isShared_908_; uint8_t v_isSharedCheck_915_; 
v___x_897_ = lean_st_ref_take(v___x_844_);
v_env_898_ = lean_ctor_get(v___x_897_, 0);
v_nextMacroScope_899_ = lean_ctor_get(v___x_897_, 1);
v_ngen_900_ = lean_ctor_get(v___x_897_, 2);
v_auxDeclNGen_901_ = lean_ctor_get(v___x_897_, 3);
v_traceState_902_ = lean_ctor_get(v___x_897_, 4);
v_messages_903_ = lean_ctor_get(v___x_897_, 6);
v_infoState_904_ = lean_ctor_get(v___x_897_, 7);
v_snapshotTasks_905_ = lean_ctor_get(v___x_897_, 8);
v_isSharedCheck_915_ = !lean_is_exclusive(v___x_897_);
if (v_isSharedCheck_915_ == 0)
{
lean_object* v_unused_916_; 
v_unused_916_ = lean_ctor_get(v___x_897_, 5);
lean_dec(v_unused_916_);
v___x_907_ = v___x_897_;
v_isShared_908_ = v_isSharedCheck_915_;
goto v_resetjp_906_;
}
else
{
lean_inc(v_snapshotTasks_905_);
lean_inc(v_infoState_904_);
lean_inc(v_messages_903_);
lean_inc(v_traceState_902_);
lean_inc(v_auxDeclNGen_901_);
lean_inc(v_ngen_900_);
lean_inc(v_nextMacroScope_899_);
lean_inc(v_env_898_);
lean_dec(v___x_897_);
v___x_907_ = lean_box(0);
v_isShared_908_ = v_isSharedCheck_915_;
goto v_resetjp_906_;
}
v_resetjp_906_:
{
uint8_t v___x_909_; lean_object* v___x_910_; lean_object* v___x_912_; 
v___x_909_ = lean_unbox(v___x_874_);
v___x_910_ = l_Lean_Kernel_enableDiag(v_env_898_, v___x_909_);
if (v_isShared_908_ == 0)
{
lean_ctor_set(v___x_907_, 5, v___x_839_);
lean_ctor_set(v___x_907_, 0, v___x_910_);
v___x_912_ = v___x_907_;
goto v_reusejp_911_;
}
else
{
lean_object* v_reuseFailAlloc_914_; 
v_reuseFailAlloc_914_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_914_, 0, v___x_910_);
lean_ctor_set(v_reuseFailAlloc_914_, 1, v_nextMacroScope_899_);
lean_ctor_set(v_reuseFailAlloc_914_, 2, v_ngen_900_);
lean_ctor_set(v_reuseFailAlloc_914_, 3, v_auxDeclNGen_901_);
lean_ctor_set(v_reuseFailAlloc_914_, 4, v_traceState_902_);
lean_ctor_set(v_reuseFailAlloc_914_, 5, v___x_839_);
lean_ctor_set(v_reuseFailAlloc_914_, 6, v_messages_903_);
lean_ctor_set(v_reuseFailAlloc_914_, 7, v_infoState_904_);
lean_ctor_set(v_reuseFailAlloc_914_, 8, v_snapshotTasks_905_);
v___x_912_ = v_reuseFailAlloc_914_;
goto v_reusejp_911_;
}
v_reusejp_911_:
{
lean_object* v___x_913_; 
v___x_913_ = lean_st_ref_set(v___x_844_, v___x_912_);
lean_inc(v___x_844_);
v___y_876_ = v___x_844_;
goto v___jp_875_;
}
}
}
else
{
lean_inc(v___x_844_);
v___y_876_ = v___x_844_;
goto v___jp_875_;
}
}
}
}
}
}
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_foldCurrFileDecls___redArg___boxed(lean_object* v_init_941_, lean_object* v_cfg_942_, lean_object* v_act_943_, lean_object* v_a_944_, lean_object* v_a_945_, lean_object* v_a_946_){
_start:
{
lean_object* v_res_947_; 
v_res_947_ = lp_mathlib_Lean_Meta_foldCurrFileDecls___redArg(v_init_941_, v_cfg_942_, v_act_943_, v_a_944_, v_a_945_);
lean_dec(v_a_945_);
lean_dec_ref(v_a_944_);
return v_res_947_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_foldCurrFileDecls(lean_object* v_00_u03b1_948_, lean_object* v_init_949_, lean_object* v_cfg_950_, lean_object* v_act_951_, lean_object* v_a_952_, lean_object* v_a_953_){
_start:
{
lean_object* v___x_955_; 
v___x_955_ = lp_mathlib_Lean_Meta_foldCurrFileDecls___redArg(v_init_949_, v_cfg_950_, v_act_951_, v_a_952_, v_a_953_);
return v___x_955_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_foldCurrFileDecls___boxed(lean_object* v_00_u03b1_956_, lean_object* v_init_957_, lean_object* v_cfg_958_, lean_object* v_act_959_, lean_object* v_a_960_, lean_object* v_a_961_, lean_object* v_a_962_){
_start:
{
lean_object* v_res_963_; 
v_res_963_ = lp_mathlib_Lean_Meta_foldCurrFileDecls(v_00_u03b1_956_, v_init_957_, v_cfg_958_, v_act_959_, v_a_960_, v_a_961_);
lean_dec(v_a_961_);
lean_dec_ref(v_a_960_);
return v_res_963_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_Lean_Meta_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Init(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Lean_FoldEnvironment(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Meta_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Lean_FoldEnvironment(uint8_t builtin) {
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
lean_object* initialize_Lean_Meta_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Init(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Lean_FoldEnvironment(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Meta_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Lean_FoldEnvironment(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Lean_FoldEnvironment(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Lean_FoldEnvironment(builtin);
}
#ifdef __cplusplus
}
#endif
