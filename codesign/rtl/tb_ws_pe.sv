`timescale 1ns/1ps
module tb_ws_pe;
    logic clk=0, rst_n=0, ce=1, load_w=0;
    logic signed [7:0] act_in=0, weight_in=0, act_out;
    logic signed [31:0] psum_in=0, psum_out;
    integer checks=0;
    logic signed [31:0] expected;
    always #5 clk=~clk;
    ws_pe dut(.*);
    task automatic edge_check(input logic signed [31:0] want);
        @(posedge clk); #1;
        if (psum_out !== want) $fatal(1,"psum got=%0d want=%0d",psum_out,want);
        checks++;
    endtask
    initial begin
        repeat(2) @(negedge clk);
        rst_n=1;
        for (int w=-128; w<=127; w++) begin
            @(negedge clk); load_w=1; weight_in=w;
            @(negedge clk); load_w=0;
            for (int a=-128; a<=127; a++) begin
                act_in=a; psum_in=17;
                expected=a*w+17;
                edge_check(expected);
                if (act_out !== act_in) $fatal(1,"activation forwarding mismatch");
                @(negedge clk);
            end
        end
        load_w=1; weight_in=1;
        @(negedge clk); load_w=0; act_in=1; psum_in=32'sh7fffffff;
        edge_check(32'sh80000000);
        @(negedge clk); ce=0; act_in=9; psum_in=123;
        edge_check(32'sh80000000);
        @(negedge clk); rst_n=0;
        edge_check(0);
        $display("PASS %0d PE checks",checks);
        $finish;
    end
endmodule
