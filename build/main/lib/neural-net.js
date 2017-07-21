"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var fs = require("fs");
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
    return Matrix;
}());
exports.Matrix = Matrix;
var NeuralNet = (function () {
    function NeuralNet(inputElements, hiddenLayers, numOutcomes, learningRate) {
        this.weightMatrices = [];
        this.numInputs = inputElements;
        this.numOutcomes = numOutcomes;
        this.lr = learningRate;
        /**
         * assemble the weights matrix [theta]
         * each individual weight is represented by theta (layerNumber) (toNumber => j) (fromNumber => k)
         * thus theta is a vector which, for any given theta besides in between the hidden and output layer, contains the number of weights equal
         * to (numinputElements + 1 (+1 because of bias node)) * numInputElements
         */
        /**
         * assembling Theta
         */
        var theta = [];
        for (var i = 1; i <= hiddenLayers; i++) {
            var layer_1 = i;
            /**
             * assembling a WeightLayer
             */
            var weightLayer_1 = Object.create([]);
            // + 2 because +1 to account for indexing and another +1 to account for biased node
            for (var k = 1; k < inputElements + 2; k++) {
                /**
                 * assembling a WeightVector
                 */
                var weightVector = Object.create([]);
                for (var j = 1; j < inputElements + 1; j++) {
                    var weightVal = Math.random();
                    weightVector.push({ value: weightVal, jVal: j, kVal: k, layer: layer_1 });
                }
                weightLayer_1.push(weightVector);
            }
            theta.push(weightLayer_1);
        }
        /**
         * now need to add the final layer of weights which are in-between the last hidden layer and the output layer
         * weight layer number is going to be the number of hidden layers + 1
         */
        var layer = hiddenLayers + 1;
        var weightLayer = Object.create([]);
        // + 2 because +1 to account for indexing and another +1 to account for biased node
        for (var k = 1; k < inputElements + 2; k++) {
            /**
             * assembling a WeightVector
             */
            var weightVector = Object.create([]);
            for (var j = 1; j < numOutcomes + 1; j++) {
                var weightVal = Math.random();
                weightVector.push({ value: weightVal, jVal: j, kVal: k, layer: layer });
            }
            weightLayer.push(weightVector);
        }
        theta.push(weightLayer);
        this._weights = theta;
        // creating the weights matrix
        /**
         * Each vector in the weights matrix will be made up of weights [
         * theta(jk), theta(j+1, k), theta(j+2, k)
         * ]
         */
        // Loop for assembling multiple matrices
        for (var i = 0; i < this._weights.length; i++) {
            var currentLayer = this._weights[i];
            var matrix = new Matrix();
            for (var _i = 0, currentLayer_1 = currentLayer; _i < currentLayer_1.length; _i++) {
                var weightArray = currentLayer_1[_i];
                var vector = [];
                for (var _a = 0, weightArray_1 = weightArray; _a < weightArray_1.length; _a++) {
                    var weight = weightArray_1[_a];
                    vector.push(weight.value);
                }
                matrix.next(vector);
            }
            this.weightMatrices.push(matrix);
        }
    }
    /**
     * Use 'layer' to choose what layer of weights you want
     * Use 'kVal' to choose from all weights that have that kVal, ie the vector that corresponds to the from node
     * Use 'jVal' to choose what toNode the weight is going to.
     * Note that when thinking about this function, you should consider the biased node ->
     * ie weightVal(1, 3, 2) will give you the weight of the first weight layer that goes from the third node of the input layer
     * to the second node of the hidden layer (but first node not including the hidden layer).
     * But, its jVal will be 1 to avoid confusion later on.
     * @param jVal
     * @param kVal
     * @param layer
     */
    NeuralNet.prototype.weightVal = function (layer, kVal, jVal) {
        var weight = this._weights[layer - 1][kVal - 1][jVal - 2];
        return weight;
    };
    /**
     *
     * @param filename Desired name of the JSON file, ie "net.json"
     * @param path Desired location of the file, ie "./output/"
     */
    NeuralNet.prototype.toJSON = function (filename, path, data) {
        var fullPath = path + filename;
        if (fs.existsSync(fullPath)) {
            fs.unlinkSync(fullPath);
            fs.writeFileSync(fullPath, data);
        }
    };
    return NeuralNet;
}());
exports.NeuralNet = NeuralNet;
