// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design internal header
// See Vmatrix_vector_mult_4x4x16.h for the primary calling header

#ifndef VERILATED_VMATRIX_VECTOR_MULT_4X4X16___024ROOT_H_
#define VERILATED_VMATRIX_VECTOR_MULT_4X4X16___024ROOT_H_  // guard

#include "verilated.h"


class Vmatrix_vector_mult_4x4x16__Syms;

class alignas(VL_CACHE_LINE_BYTES) Vmatrix_vector_mult_4x4x16___024root final : public VerilatedModule {
  public:

    // DESIGN SPECIFIC STATE
    VL_IN8(clk,0,0);
    VL_IN8(reset,0,0);
    VL_IN8(enable,0,0);
    CData/*0:0*/ __VstlFirstIteration;
    CData/*0:0*/ __VicoFirstIteration;
    CData/*0:0*/ __Vtrigprevexpr___TOP__clk__0;
    CData/*0:0*/ __VactContinue;
    IData/*31:0*/ matrix_vector_mult_4x4x16__DOT____Vcellout__ipu0__result;
    IData/*31:0*/ matrix_vector_mult_4x4x16__DOT____Vcellout__ipu1__result;
    IData/*31:0*/ matrix_vector_mult_4x4x16__DOT____Vcellout__ipu2__result;
    IData/*31:0*/ matrix_vector_mult_4x4x16__DOT____Vcellout__ipu3__result;
    IData/*31:0*/ matrix_vector_mult_4x4x16__DOT__ipu0__DOT__i;
    IData/*31:0*/ matrix_vector_mult_4x4x16__DOT__ipu1__DOT__i;
    IData/*31:0*/ matrix_vector_mult_4x4x16__DOT__ipu2__DOT__i;
    IData/*31:0*/ matrix_vector_mult_4x4x16__DOT__ipu3__DOT__i;
    IData/*31:0*/ __VactIterCount;
    VL_IN16(activations[4],15,0);
    VL_IN16(weights[4][4],15,0);
    VL_OUT(results[4],31,0);
    VlUnpacked<SData/*15:0*/, 4> matrix_vector_mult_4x4x16__DOT____Vcellinp__ipu0__weights;
    VlUnpacked<SData/*15:0*/, 4> matrix_vector_mult_4x4x16__DOT____Vcellinp__ipu1__weights;
    VlUnpacked<SData/*15:0*/, 4> matrix_vector_mult_4x4x16__DOT____Vcellinp__ipu2__weights;
    VlUnpacked<SData/*15:0*/, 4> matrix_vector_mult_4x4x16__DOT____Vcellinp__ipu3__weights;
    VlUnpacked<SData/*15:0*/, 4> matrix_vector_mult_4x4x16__DOT__ipu0__DOT__activations_reg;
    VlUnpacked<SData/*15:0*/, 4> matrix_vector_mult_4x4x16__DOT__ipu0__DOT__weights_reg;
    VlUnpacked<SData/*15:0*/, 4> matrix_vector_mult_4x4x16__DOT__ipu1__DOT__activations_reg;
    VlUnpacked<SData/*15:0*/, 4> matrix_vector_mult_4x4x16__DOT__ipu1__DOT__weights_reg;
    VlUnpacked<SData/*15:0*/, 4> matrix_vector_mult_4x4x16__DOT__ipu2__DOT__activations_reg;
    VlUnpacked<SData/*15:0*/, 4> matrix_vector_mult_4x4x16__DOT__ipu2__DOT__weights_reg;
    VlUnpacked<SData/*15:0*/, 4> matrix_vector_mult_4x4x16__DOT__ipu3__DOT__activations_reg;
    VlUnpacked<SData/*15:0*/, 4> matrix_vector_mult_4x4x16__DOT__ipu3__DOT__weights_reg;
    VlUnpacked<CData/*0:0*/, 3> __Vm_traceActivity;
    VlTriggerVec<1> __VstlTriggered;
    VlTriggerVec<1> __VicoTriggered;
    VlTriggerVec<1> __VactTriggered;
    VlTriggerVec<1> __VnbaTriggered;

    // INTERNAL VARIABLES
    Vmatrix_vector_mult_4x4x16__Syms* const vlSymsp;

    // CONSTRUCTORS
    Vmatrix_vector_mult_4x4x16___024root(Vmatrix_vector_mult_4x4x16__Syms* symsp, const char* v__name);
    ~Vmatrix_vector_mult_4x4x16___024root();
    VL_UNCOPYABLE(Vmatrix_vector_mult_4x4x16___024root);

    // INTERNAL METHODS
    void __Vconfigure(bool first);
};


#endif  // guard
