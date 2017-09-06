/**
 * Always listed from top to bottom, ie [1,2,3] is equivalent to
 * [1
 * 2
 * 3]
 */
export type Vector = number[];

export interface AugmentedVector {
    vector: Vector;
    /**
     * what column number, USING non-zero indexing (first is '1')
     */
    posInMatrix: number;
}
/**
 * Column based 'vector-based'
 * IE the matrix made up of two vectors [1,2,3] and [4,5,6] really looks like
 * 1 4
 * 2 5
 * 3 6
 */
export interface MatrixI {
    postitionedVectors: AugmentedVector[];
    next(vector: Vector): void;
}

export class Matrix implements MatrixI {
    postitionedVectors: AugmentedVector[];
    public next(inVector: Vector) {
        this.postitionedVectors.push(
            { vector: inVector, posInMatrix: this.postitionedVectors.length + 1 }
        );
    }
    constructor(inVector?: Vector) {
        if (inVector) {
            this.postitionedVectors = [{ vector: inVector, posInMatrix: 1 }];
        }
        else {
            this.postitionedVectors = [];
        }
    }
    get vectors() {
        let realVectors: Vector[] = [];
        for (let vector of this.postitionedVectors) {
            realVectors.push(vector.vector);
        }
        return realVectors;
    }
    toRowFormat(): number[][] {
        const elements: number[][] = [];
        const vectors = this.vectors;
        const cols = vectors[0].length;
        const rows = vectors.length;

        for (let i = 0; i < cols; i++) {
            const row: number[] = [];

            for (let j = 0; j < rows; j++) {
                row.push(vectors[j][i]);
            }
            elements.push(row);
        }
        return elements;
    }
}

export function dotProduct(lhs: number[], rhs: number[]) {
    if (lhs.length === rhs.length) {
        const dots: number[] = [];
        let accumulator = 0;
        for (let i = 0; i < lhs.length; i++) {
            const lhsEl = lhs[i];
            const rhsEl = rhs[i];
            const dot = lhsEl * rhsEl;
            accumulator = accumulator + dot;
        }
        return accumulator;
    }
    else {
        throw new Error(`Can't execute dot product, lhs length = ${lhs.length} and rhs length = ${rhs.length}`);
    }
}
