import { AugmentedVector, dotProduct, Matrix, MatrixI, Vector } from './net-matrix';
import * as assert from 'assert';

describe("Matrix testing", () => {
    let matrix: Matrix = new Matrix();
    it("Rows to Columns", () => {
        matrix.next([1,2,3]);
        matrix.next([4,5,6]);
        matrix.next([7,8,9]);
        assert.deepEqual(matrix.toRowFormat(), [[1,4,7],[2,5,8], [3,6,9]]);
    });
});
