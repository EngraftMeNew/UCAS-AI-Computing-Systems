// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design internal header
// See Vmatrix_mult_4x4x4x16.h for the primary calling header

#ifndef VERILATED_VMATRIX_MULT_4X4X4X16___024ROOT_H_
#define VERILATED_VMATRIX_MULT_4X4X4X16___024ROOT_H_  // guard

#include "verilated.h"


class Vmatrix_mult_4x4x4x16__Syms;

class alignas(VL_CACHE_LINE_BYTES) Vmatrix_mult_4x4x4x16___024root final : public VerilatedModule {
  public:

    // DESIGN SPECIFIC STATE
    // Anonymous structures to workaround compiler member-count bugs
    struct {
        VL_IN8(clk,0,0);
        VL_IN8(reset,0,0);
        VL_IN8(enable,0,0);
        CData/*0:0*/ __VstlFirstIteration;
        CData/*0:0*/ __VicoFirstIteration;
        CData/*0:0*/ __Vtrigprevexpr___TOP__clk__0;
        CData/*0:0*/ __VactContinue;
        IData/*31:0*/ matrix_mult_4x4x4x16__DOT__mxv0__DOT____Vcellout__ipu0__result;
        IData/*31:0*/ matrix_mult_4x4x4x16__DOT__mxv0__DOT____Vcellout__ipu1__result;
        IData/*31:0*/ matrix_mult_4x4x4x16__DOT__mxv0__DOT____Vcellout__ipu2__result;
        IData/*31:0*/ matrix_mult_4x4x4x16__DOT__mxv0__DOT____Vcellout__ipu3__result;
        IData/*31:0*/ matrix_mult_4x4x4x16__DOT__mxv0__DOT__ipu0__DOT__i;
        IData/*31:0*/ matrix_mult_4x4x4x16__DOT__mxv0__DOT__ipu1__DOT__i;
        IData/*31:0*/ matrix_mult_4x4x4x16__DOT__mxv0__DOT__ipu2__DOT__i;
        IData/*31:0*/ matrix_mult_4x4x4x16__DOT__mxv0__DOT__ipu3__DOT__i;
        IData/*31:0*/ matrix_mult_4x4x4x16__DOT__mxv1__DOT____Vcellout__ipu0__result;
        IData/*31:0*/ matrix_mult_4x4x4x16__DOT__mxv1__DOT____Vcellout__ipu1__result;
        IData/*31:0*/ matrix_mult_4x4x4x16__DOT__mxv1__DOT____Vcellout__ipu2__result;
        IData/*31:0*/ matrix_mult_4x4x4x16__DOT__mxv1__DOT____Vcellout__ipu3__result;
        IData/*31:0*/ matrix_mult_4x4x4x16__DOT__mxv1__DOT__ipu0__DOT__i;
        IData/*31:0*/ matrix_mult_4x4x4x16__DOT__mxv1__DOT__ipu1__DOT__i;
        IData/*31:0*/ matrix_mult_4x4x4x16__DOT__mxv1__DOT__ipu2__DOT__i;
        IData/*31:0*/ matrix_mult_4x4x4x16__DOT__mxv1__DOT__ipu3__DOT__i;
        IData/*31:0*/ matrix_mult_4x4x4x16__DOT__mxv2__DOT____Vcellout__ipu0__result;
        IData/*31:0*/ matrix_mult_4x4x4x16__DOT__mxv2__DOT____Vcellout__ipu1__result;
        IData/*31:0*/ matrix_mult_4x4x4x16__DOT__mxv2__DOT____Vcellout__ipu2__result;
        IData/*31:0*/ matrix_mult_4x4x4x16__DOT__mxv2__DOT____Vcellout__ipu3__result;
        IData/*31:0*/ matrix_mult_4x4x4x16__DOT__mxv2__DOT__ipu0__DOT__i;
        IData/*31:0*/ matrix_mult_4x4x4x16__DOT__mxv2__DOT__ipu1__DOT__i;
        IData/*31:0*/ matrix_mult_4x4x4x16__DOT__mxv2__DOT__ipu2__DOT__i;
        IData/*31:0*/ matrix_mult_4x4x4x16__DOT__mxv2__DOT__ipu3__DOT__i;
        IData/*31:0*/ matrix_mult_4x4x4x16__DOT__mxv3__DOT____Vcellout__ipu0__result;
        IData/*31:0*/ matrix_mult_4x4x4x16__DOT__mxv3__DOT____Vcellout__ipu1__result;
        IData/*31:0*/ matrix_mult_4x4x4x16__DOT__mxv3__DOT____Vcellout__ipu2__result;
        IData/*31:0*/ matrix_mult_4x4x4x16__DOT__mxv3__DOT____Vcellout__ipu3__result;
        IData/*31:0*/ matrix_mult_4x4x4x16__DOT__mxv3__DOT__ipu0__DOT__i;
        IData/*31:0*/ matrix_mult_4x4x4x16__DOT__mxv3__DOT__ipu1__DOT__i;
        IData/*31:0*/ matrix_mult_4x4x4x16__DOT__mxv3__DOT__ipu2__DOT__i;
        IData/*31:0*/ matrix_mult_4x4x4x16__DOT__mxv3__DOT__ipu3__DOT__i;
        IData/*31:0*/ __VactIterCount;
        VL_IN16(activations[4][4],15,0);
        VL_IN16(weights[4][4],15,0);
        VL_OUT(results[4][4],31,0);
        VlUnpacked<IData/*31:0*/, 4> matrix_mult_4x4x4x16__DOT____Vcellout__mxv0__results;
        VlUnpacked<SData/*15:0*/, 4> matrix_mult_4x4x4x16__DOT____Vcellinp__mxv0__activations;
        VlUnpacked<IData/*31:0*/, 4> matrix_mult_4x4x4x16__DOT____Vcellout__mxv1__results;
        VlUnpacked<SData/*15:0*/, 4> matrix_mult_4x4x4x16__DOT____Vcellinp__mxv1__activations;
        VlUnpacked<IData/*31:0*/, 4> matrix_mult_4x4x4x16__DOT____Vcellout__mxv2__results;
        VlUnpacked<SData/*15:0*/, 4> matrix_mult_4x4x4x16__DOT____Vcellinp__mxv2__activations;
        VlUnpacked<IData/*31:0*/, 4> matrix_mult_4x4x4x16__DOT____Vcellout__mxv3__results;
        VlUnpacked<SData/*15:0*/, 4> matrix_mult_4x4x4x16__DOT____Vcellinp__mxv3__activations;
        VlUnpacked<SData/*15:0*/, 4> matrix_mult_4x4x4x16__DOT__mxv0__DOT____Vcellinp__ipu0__weights;
        VlUnpacked<SData/*15:0*/, 4> matrix_mult_4x4x4x16__DOT__mxv0__DOT____Vcellinp__ipu1__weights;
        VlUnpacked<SData/*15:0*/, 4> matrix_mult_4x4x4x16__DOT__mxv0__DOT____Vcellinp__ipu2__weights;
        VlUnpacked<SData/*15:0*/, 4> matrix_mult_4x4x4x16__DOT__mxv0__DOT____Vcellinp__ipu3__weights;
        VlUnpacked<SData/*15:0*/, 4> matrix_mult_4x4x4x16__DOT__mxv0__DOT__ipu0__DOT__activations_reg;
        VlUnpacked<SData/*15:0*/, 4> matrix_mult_4x4x4x16__DOT__mxv0__DOT__ipu0__DOT__weights_reg;
        VlUnpacked<SData/*15:0*/, 4> matrix_mult_4x4x4x16__DOT__mxv0__DOT__ipu1__DOT__activations_reg;
        VlUnpacked<SData/*15:0*/, 4> matrix_mult_4x4x4x16__DOT__mxv0__DOT__ipu1__DOT__weights_reg;
        VlUnpacked<SData/*15:0*/, 4> matrix_mult_4x4x4x16__DOT__mxv0__DOT__ipu2__DOT__activations_reg;
        VlUnpacked<SData/*15:0*/, 4> matrix_mult_4x4x4x16__DOT__mxv0__DOT__ipu2__DOT__weights_reg;
        VlUnpacked<SData/*15:0*/, 4> matrix_mult_4x4x4x16__DOT__mxv0__DOT__ipu3__DOT__activations_reg;
        VlUnpacked<SData/*15:0*/, 4> matrix_mult_4x4x4x16__DOT__mxv0__DOT__ipu3__DOT__weights_reg;
        VlUnpacked<SData/*15:0*/, 4> matrix_mult_4x4x4x16__DOT__mxv1__DOT____Vcellinp__ipu0__weights;
    };
    struct {
        VlUnpacked<SData/*15:0*/, 4> matrix_mult_4x4x4x16__DOT__mxv1__DOT____Vcellinp__ipu1__weights;
        VlUnpacked<SData/*15:0*/, 4> matrix_mult_4x4x4x16__DOT__mxv1__DOT____Vcellinp__ipu2__weights;
        VlUnpacked<SData/*15:0*/, 4> matrix_mult_4x4x4x16__DOT__mxv1__DOT____Vcellinp__ipu3__weights;
        VlUnpacked<SData/*15:0*/, 4> matrix_mult_4x4x4x16__DOT__mxv1__DOT__ipu0__DOT__activations_reg;
        VlUnpacked<SData/*15:0*/, 4> matrix_mult_4x4x4x16__DOT__mxv1__DOT__ipu0__DOT__weights_reg;
        VlUnpacked<SData/*15:0*/, 4> matrix_mult_4x4x4x16__DOT__mxv1__DOT__ipu1__DOT__activations_reg;
        VlUnpacked<SData/*15:0*/, 4> matrix_mult_4x4x4x16__DOT__mxv1__DOT__ipu1__DOT__weights_reg;
        VlUnpacked<SData/*15:0*/, 4> matrix_mult_4x4x4x16__DOT__mxv1__DOT__ipu2__DOT__activations_reg;
        VlUnpacked<SData/*15:0*/, 4> matrix_mult_4x4x4x16__DOT__mxv1__DOT__ipu2__DOT__weights_reg;
        VlUnpacked<SData/*15:0*/, 4> matrix_mult_4x4x4x16__DOT__mxv1__DOT__ipu3__DOT__activations_reg;
        VlUnpacked<SData/*15:0*/, 4> matrix_mult_4x4x4x16__DOT__mxv1__DOT__ipu3__DOT__weights_reg;
        VlUnpacked<SData/*15:0*/, 4> matrix_mult_4x4x4x16__DOT__mxv2__DOT____Vcellinp__ipu0__weights;
        VlUnpacked<SData/*15:0*/, 4> matrix_mult_4x4x4x16__DOT__mxv2__DOT____Vcellinp__ipu1__weights;
        VlUnpacked<SData/*15:0*/, 4> matrix_mult_4x4x4x16__DOT__mxv2__DOT____Vcellinp__ipu2__weights;
        VlUnpacked<SData/*15:0*/, 4> matrix_mult_4x4x4x16__DOT__mxv2__DOT____Vcellinp__ipu3__weights;
        VlUnpacked<SData/*15:0*/, 4> matrix_mult_4x4x4x16__DOT__mxv2__DOT__ipu0__DOT__activations_reg;
        VlUnpacked<SData/*15:0*/, 4> matrix_mult_4x4x4x16__DOT__mxv2__DOT__ipu0__DOT__weights_reg;
        VlUnpacked<SData/*15:0*/, 4> matrix_mult_4x4x4x16__DOT__mxv2__DOT__ipu1__DOT__activations_reg;
        VlUnpacked<SData/*15:0*/, 4> matrix_mult_4x4x4x16__DOT__mxv2__DOT__ipu1__DOT__weights_reg;
        VlUnpacked<SData/*15:0*/, 4> matrix_mult_4x4x4x16__DOT__mxv2__DOT__ipu2__DOT__activations_reg;
        VlUnpacked<SData/*15:0*/, 4> matrix_mult_4x4x4x16__DOT__mxv2__DOT__ipu2__DOT__weights_reg;
        VlUnpacked<SData/*15:0*/, 4> matrix_mult_4x4x4x16__DOT__mxv2__DOT__ipu3__DOT__activations_reg;
        VlUnpacked<SData/*15:0*/, 4> matrix_mult_4x4x4x16__DOT__mxv2__DOT__ipu3__DOT__weights_reg;
        VlUnpacked<SData/*15:0*/, 4> matrix_mult_4x4x4x16__DOT__mxv3__DOT____Vcellinp__ipu0__weights;
        VlUnpacked<SData/*15:0*/, 4> matrix_mult_4x4x4x16__DOT__mxv3__DOT____Vcellinp__ipu1__weights;
        VlUnpacked<SData/*15:0*/, 4> matrix_mult_4x4x4x16__DOT__mxv3__DOT____Vcellinp__ipu2__weights;
        VlUnpacked<SData/*15:0*/, 4> matrix_mult_4x4x4x16__DOT__mxv3__DOT____Vcellinp__ipu3__weights;
        VlUnpacked<SData/*15:0*/, 4> matrix_mult_4x4x4x16__DOT__mxv3__DOT__ipu0__DOT__activations_reg;
        VlUnpacked<SData/*15:0*/, 4> matrix_mult_4x4x4x16__DOT__mxv3__DOT__ipu0__DOT__weights_reg;
        VlUnpacked<SData/*15:0*/, 4> matrix_mult_4x4x4x16__DOT__mxv3__DOT__ipu1__DOT__activations_reg;
        VlUnpacked<SData/*15:0*/, 4> matrix_mult_4x4x4x16__DOT__mxv3__DOT__ipu1__DOT__weights_reg;
        VlUnpacked<SData/*15:0*/, 4> matrix_mult_4x4x4x16__DOT__mxv3__DOT__ipu2__DOT__activations_reg;
        VlUnpacked<SData/*15:0*/, 4> matrix_mult_4x4x4x16__DOT__mxv3__DOT__ipu2__DOT__weights_reg;
        VlUnpacked<SData/*15:0*/, 4> matrix_mult_4x4x4x16__DOT__mxv3__DOT__ipu3__DOT__activations_reg;
        VlUnpacked<SData/*15:0*/, 4> matrix_mult_4x4x4x16__DOT__mxv3__DOT__ipu3__DOT__weights_reg;
        VlUnpacked<CData/*0:0*/, 3> __Vm_traceActivity;
    };
    VlTriggerVec<1> __VstlTriggered;
    VlTriggerVec<1> __VicoTriggered;
    VlTriggerVec<1> __VactTriggered;
    VlTriggerVec<1> __VnbaTriggered;

    // INTERNAL VARIABLES
    Vmatrix_mult_4x4x4x16__Syms* const vlSymsp;

    // CONSTRUCTORS
    Vmatrix_mult_4x4x4x16___024root(Vmatrix_mult_4x4x4x16__Syms* symsp, const char* v__name);
    ~Vmatrix_mult_4x4x4x16___024root();
    VL_UNCOPYABLE(Vmatrix_mult_4x4x4x16___024root);

    // INTERNAL METHODS
    void __Vconfigure(bool first);
};


#endif  // guard
