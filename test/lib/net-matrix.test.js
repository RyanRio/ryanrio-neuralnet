"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var net_matrix_1 = require("./net-matrix");
var assert = require("assert");
describe("Matrix testing", function () {
    var matrix = new net_matrix_1.Matrix();
    it("Rows to Columns", function () {
        matrix.next([1, 2, 3]);
        matrix.next([4, 5, 6]);
        matrix.next([7, 8, 9]);
        assert.deepEqual(matrix.toRowFormat(), [[1, 4, 7], [2, 5, 8], [3, 6, 9]]);
    });
});
