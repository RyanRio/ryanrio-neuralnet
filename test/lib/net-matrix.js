"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var Matrix = (function () {
    function Matrix(inVector) {
        if (inVector) {
            this.postitionedVectors = [{ vector: inVector, posInMatrix: 1 }];
        }
        else {
            this.postitionedVectors = [];
        }
    }
    Matrix.prototype.next = function (inVector) {
        this.postitionedVectors.push({ vector: inVector, posInMatrix: this.postitionedVectors.length + 1 });
    };
    Object.defineProperty(Matrix.prototype, "vectors", {
        get: function () {
            var realVectors = [];
            for (var _i = 0, _a = this.postitionedVectors; _i < _a.length; _i++) {
                var vector = _a[_i];
                realVectors.push(vector.vector);
            }
            return realVectors;
        },
        enumerable: true,
        configurable: true
    });
    Matrix.prototype.toRowFormat = function () {
        var elements = [];
        var vectors = this.vectors;
        var cols = vectors[0].length;
        var rows = vectors.length;
        for (var i = 0; i < cols; i++) {
            var row = [];
            for (var j = 0; j < rows; j++) {
                row.push(vectors[j][i]);
            }
            elements.push(row);
        }
        return elements;
    };
    return Matrix;
}());
exports.Matrix = Matrix;
function dotProduct(lhs, rhs) {
    if (lhs.length === rhs.length) {
        var dots = [];
        var accumulator = 0;
        for (var i = 0; i < lhs.length; i++) {
            var lhsEl = lhs[i];
            var rhsEl = rhs[i];
            var dot = lhsEl * rhsEl;
            accumulator = accumulator + dot;
        }
        return accumulator;
    }
    else {
        throw new Error("Can't execute dot product, lhs length = " + lhs.length + " and rhs length = " + rhs.length);
    }
}
exports.dotProduct = dotProduct;
