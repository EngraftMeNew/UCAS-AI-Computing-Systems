// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design internal header
// See Vinner_product_4x16.h for the primary calling header

#ifndef VERILATED_VINNER_PRODUCT_4X16___024ROOT_H_
#define VERILATED_VINNER_PRODUCT_4X16___024ROOT_H_  // guard

#include "verilated.h"


class Vinner_product_4x16__Syms;

class alignas(VL_CACHE_LINE_BYTES) Vinner_product_4x16___024root final : public VerilatedModule {
  public:

    // DESIGN SPECIFIC STATE
    VL_IN8(clk,0,0);
    VL_IN8(reset,0,0);
    VL_IN8(enable,0,0);
    CData/*0:0*/ __Vtrigprevexpr___TOP__clk__0;
    CData/*0:0*/ __VactContinue;
    VL_OUT(result,31,0);
    IData/*31:0*/ inner_product_4x16__DOT__i;
    IData/*31:0*/ __VactIterCount;
    VL_IN16(activations[4],15,0);
    VL_IN16(weights[4],15,0);
    VlUnpacked<SData/*15:0*/, 4> inner_product_4x16__DOT__activations_reg;
    VlUnpacked<SData/*15:0*/, 4> inner_product_4x16__DOT__weights_reg;
    VlUnpacked<CData/*0:0*/, 2> __Vm_traceActivity;
    VlTriggerVec<1> __VactTriggered;
    VlTriggerVec<1> __VnbaTriggered;

    // INTERNAL VARIABLES
    Vinner_product_4x16__Syms* const vlSymsp;

    // CONSTRUCTORS
    Vinner_product_4x16___024root(Vinner_product_4x16__Syms* symsp, const char* v__name);
    ~Vinner_product_4x16___024root();
    VL_UNCOPYABLE(Vinner_product_4x16___024root);

    // INTERNAL METHODS
    void __Vconfigure(bool first);
};


#endif  // guard
