// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Symbol table internal header
//
// Internal details; most calling programs do not need this header,
// unless using verilator public meta comments.

#ifndef VERILATED_VMATRIX_VECTOR_MULT_4X4X16__SYMS_H_
#define VERILATED_VMATRIX_VECTOR_MULT_4X4X16__SYMS_H_  // guard

#include "verilated.h"

// INCLUDE MODEL CLASS

#include "Vmatrix_vector_mult_4x4x16.h"

// INCLUDE MODULE CLASSES
#include "Vmatrix_vector_mult_4x4x16___024root.h"

// SYMS CLASS (contains all model state)
class alignas(VL_CACHE_LINE_BYTES)Vmatrix_vector_mult_4x4x16__Syms final : public VerilatedSyms {
  public:
    // INTERNAL STATE
    Vmatrix_vector_mult_4x4x16* const __Vm_modelp;
    bool __Vm_activity = false;  ///< Used by trace routines to determine change occurred
    uint32_t __Vm_baseCode = 0;  ///< Used by trace routines when tracing multiple models
    VlDeleter __Vm_deleter;
    bool __Vm_didInit = false;

    // MODULE INSTANCE STATE
    Vmatrix_vector_mult_4x4x16___024root TOP;

    // CONSTRUCTORS
    Vmatrix_vector_mult_4x4x16__Syms(VerilatedContext* contextp, const char* namep, Vmatrix_vector_mult_4x4x16* modelp);
    ~Vmatrix_vector_mult_4x4x16__Syms();

    // METHODS
    const char* name() { return TOP.name(); }
};

#endif  // guard
