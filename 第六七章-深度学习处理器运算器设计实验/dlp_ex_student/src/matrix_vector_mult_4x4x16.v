module matrix_vector_mult_4x4x16 (
    input wire clk,
    input wire reset,
    input wire enable,
    input wire signed [15:0] activations [0:3],
    input wire signed [15:0] weights [0:3][0:3],
    output wire signed [31:0] results [0:3]
);

inner_product_4x16 ipu0 (
    .clk(clk),
    .reset(reset),
    .enable(enable),
    .activations(activations),
    .weights(weights[0]),
    .result(results[0])
);
inner_product_4x16 ipu1 (
    .clk(clk),
    .reset(reset),
    .enable(enable),
    .activations(activations),
    .weights(weights[1]),
    .result(results[1])
);
inner_product_4x16 ipu2 (
    .clk(clk),
    .reset(reset),
    .enable(enable),
    .activations(activations),
    .weights(weights[2]),
    .result(results[2])
);
inner_product_4x16 ipu3 (
    .clk(clk),
    .reset(reset),
    .enable(enable),
    .activations(activations),
    .weights(weights[3]),
    .result(results[3])
);

endmodule
