"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var NeuralNet = (function () {
    /**
     * Creates a NeuralNet which has a JSON representation
     * @param net
     */
    function NeuralNet(net) {
        this.prePreppedNet = net;
        this.preppedNet = this.prepJSON(this.prePreppedNet);
        this.JSON = JSON.stringify(this.preppedNet);
    }
    /**
     * Changes the prePreppedNet and then update the PreppedNet and the JSON
     * @param optionalNet Object which contains what you want to change about the prePreppedNet
     */
    NeuralNet.prototype.adjustNet = function (optionalNet) {
        var weights = optionalNet.weights;
        var tVals = optionalNet.targetValues;
        var inputs = optionalNet.inputLayer;
        var hiddenLayers = optionalNet.hiddenLayers;
        if (weights) {
            this.prePreppedNet.weights = weights;
        }
        if (tVals) {
            this.prePreppedNet.targetValues = tVals;
        }
        if (inputs) {
            this.prePreppedNet.inputLayer = inputs;
        }
        if (hiddenLayers) {
            this.prePreppedNet.hiddenLayers = hiddenLayers;
        }
        this.update();
    };
    /**
     * Updates the preppedNet and JSON using the current prePreppedNet
     */
    NeuralNet.prototype.update = function () {
        this.preppedNet = this.prepJSON(this.prePreppedNet);
        this.JSON = JSON.stringify(this.preppedNet);
    };
    NeuralNet.prototype.dealHidden = function (layers) {
        var tensors = [];
        for (var _i = 0, layers_1 = layers; _i < layers_1.length; _i++) {
            var layer = layers_1[_i];
            var size = [layer.rows, layer.cols];
            var elements = this.getElements(layer);
            tensors.push({ size: size, elements: elements });
        }
        return tensors;
    };
    NeuralNet.prototype.dealWeights = function (weights) {
        var size = [weights.rows, weights.cols];
        var elements = this.getElements(weights);
        return { size: size, elements: elements };
    };
    NeuralNet.prototype.dealTVal = function (tvals) {
        var size = [tvals.rows, tvals.cols];
        var elements = this.getElements(tvals);
        return { size: size, elements: elements };
    };
    NeuralNet.prototype.dealInput = function (input) {
        var size = [input.rows, input.cols];
        var elements = this.getElements(input);
        return { size: size, elements: elements };
    };
    NeuralNet.prototype.getElements = function (matrix) {
        var elements = [];
        for (var i = 0; i < matrix.rows; i++) {
            elements.push([]);
            for (var j = 0; j < matrix.cols; j++) {
                elements[i].push(matrix.getElement(i + 1, j + 1));
            }
        }
        return elements;
    };
    /**
     * Change the prepreppednet to a JSON-stringify compatible format
     * @param net
     */
    NeuralNet.prototype.prepJSON = function (net) {
        var weightsJSON = this.dealWeights(net.weights);
        var targetValuesJSON = this.dealTVal(net.targetValues);
        var inputLayerJSON = this.dealInput(net.inputLayer);
        var hiddenLayersJSON = this.dealHidden(net.hiddenLayers);
        var weights = weightsJSON;
        var targets = targetValuesJSON;
        var inputs = inputLayerJSON;
        var hiddenLayers = hiddenLayersJSON;
        return {
            weights: weights,
            targets: targets,
            inputs: inputs,
            hiddenLayers: hiddenLayers
        };
    };
    return NeuralNet;
}());
exports.NeuralNet = NeuralNet;
