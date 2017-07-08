"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var matrix_1 = require("@ryanrio/matrix");
/**
 * @param weights The weights that should be changed
 * @param targetValues The targets that should be changed
 * @param inputLayer the inputLayer values that should be changed
 * @param hiddenLayers the hiddenLayer(s) values that should be changed
 */
var OptionalNet = (function () {
    function OptionalNet(weights, tVals, inputs, hiddenLayers) {
        if (weights) {
            this.weights = matrix_1.matrix(weights);
        }
        if (tVals) {
            this.targetValues = matrix_1.matrix([tVals]);
        }
        if (inputs) {
            this.inputLayer = matrix_1.matrix([inputs]);
        }
        if (hiddenLayers) {
            this.hiddenLayers = [matrix_1.matrix([hiddenLayers])];
        }
    }
    return OptionalNet;
}());
var NetNode = (function () {
    function NetNode(Value) {
        this.Value = Value;
    }
    return NetNode;
}());
var Layer = (function () {
    function Layer(layer, xNodes, yNodes, Sl) {
        this.layer = layer;
        this.xNodes = xNodes;
        this.Sl = Sl;
        this.yNodes = yNodes;
    }
    return Layer;
}());
var NetSpec = (function () {
    function NetSpec(input, hidden, output) {
        this.n_inputNodes = input;
        this.n_hiddenNodes = hidden;
        this.n_outputNodes = output;
    }
    return NetSpec;
}());
exports.NetSpec = NetSpec;
var NeuralNet = (function () {
    /**
     * Creates a NeuralNet which has a JSON representation
     * @param net
     */
    function NeuralNet(net) {
        if (this.isPrePreppedNet(net)) {
            console.log("PrePreppedNet");
            this.prePreppedNet = net;
            this.preppedNet = this.prepJSON(this.prePreppedNet);
            this.JSON = JSON.stringify(this.preppedNet);
        }
        else {
            var genArray = function (iterator) {
                var array = [];
                for (var i = 0; i < iterator; i++) {
                    array.push(Math.random());
                }
                return array;
            };
            var weights = matrix_1.matrix([[]]);
            for (var i = 0; i < net.n_hiddenNodes; i++) {
                if (i === 0) {
                    weights.elements[0] = genArray(net.n_inputNodes + 1);
                }
                else {
                    weights.elements.push(genArray(net.n_inputNodes + 1));
                }
            }
            for (var i = 0; i < net.n_hiddenNodes; i++) {
                weights.elements.push(genArray(net.n_outputNodes));
            }
            var inputs = matrix_1.matrix([[]]);
            var e = [];
            for (var i = 0; i < net.n_inputNodes + 1; i++) {
                e.push(1.0);
            }
            inputs.elements[0] = e;
            var outputs = matrix_1.matrix([[]]);
            outputs.elements[0] = e;
            this.prePreppedNet = {
                inputLayer: matrix_1.matrix(inputs.elements),
                weights: matrix_1.matrix(weights.elements),
                targetValues: matrix_1.matrix(outputs.elements),
                hiddenLayers: [matrix_1.matrix([[]])]
            };
            console.log(this.prePreppedNet.weights.rows);
            this.preppedNet = this.prepJSON(this.prePreppedNet);
            this.JSON = JSON.stringify(this.preppedNet);
        }
        this.initiateNet();
    }
    NeuralNet.prototype.isPrePreppedNet = function (object) {
        return 'member' in object;
    };
    NeuralNet.prototype.initiateNet = function () {
        var net = this.preppedNet;
        // input
        var xNodes = this.constructNodes(net.inputs);
        var inputLayerNum = 0;
        var inputSl = net.inputs.elements[0].length;
        var yNodes = this.activate(xNodes);
        var inputLayer = new Layer(inputLayerNum, xNodes, yNodes, inputSl);
        // Hidden Layer
        // xNodes of hidden layer, for now assume there is only one hidden layer
        var xNodesH = this.sumFromWeights(net.weights.elements[0], yNodes);
        var hiddenLayerNum = 1;
        var hiddenSl = net.hiddenLayers[0].elements[0].length;
        var yNodesH = this.activate(xNodesH);
        var hiddenLayer = new Layer(hiddenLayerNum, xNodesH, yNodesH, hiddenSl);
        // Output Layer
        var xNodesO = this.sumFromWeights(net.weights.elements[1], yNodesH);
        var outputLayerNum = 2;
        var outputSl = net.targets.elements[0].length;
        var yNodesO = this.activate(xNodesO);
        var outputLayer = new Layer(outputLayerNum, xNodesH, yNodesH, hiddenSl);
        var inputElements = [];
        for (var _i = 0, yNodes_1 = yNodes; _i < yNodes_1.length; _i++) {
            var node = yNodes_1[_i];
            inputElements.push(node.Value);
        }
        var hiddenElements = [];
        for (var _a = 0, yNodesH_1 = yNodesH; _a < yNodesH_1.length; _a++) {
            var node = yNodesH_1[_a];
            hiddenElements.push(node.Value);
        }
        var outputElements = [];
        for (var _b = 0, yNodesO_1 = yNodesO; _b < yNodesO_1.length; _b++) {
            var node = yNodesO_1[_b];
            outputElements.push(node.Value);
        }
        this.adjustNet(new OptionalNet(undefined, outputElements, inputElements, hiddenElements));
    };
    NeuralNet.prototype.sumFromWeights = function (weights, yNodes) {
        var returnArray = [];
        for (var k = 0; k < yNodes.length; k++) {
            var sum = 0;
            for (var _i = 0, weights_1 = weights; _i < weights_1.length; _i++) {
                var weight = weights_1[_i];
                sum = weight * yNodes[k].Value;
            }
            returnArray.push(new NetNode(sum));
        }
        return returnArray;
    };
    NeuralNet.prototype.activate = function (activatable) {
        var returnArray = [];
        for (var _i = 0, activatable_1 = activatable; _i < activatable_1.length; _i++) {
            var node = activatable_1[_i];
            returnArray.push(new NetNode((1 / (1 + Math.exp(-node.Value)))));
        }
        return returnArray;
    };
    NeuralNet.prototype.constructNodes = function (input) {
        var returnArray = [];
        for (var _i = 0, _a = input.elements[0]; _i < _a.length; _i++) {
            var element = _a[_i];
            returnArray.push(new NetNode(element));
        }
        return returnArray;
    };
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
            var elements = layer.elements;
            tensors.push({ size: size, elements: elements });
        }
        return tensors;
    };
    NeuralNet.prototype.dealWeights = function (weights) {
        var size = [weights.rows, weights.cols];
        var elements = weights.elements;
        return { size: size, elements: elements };
    };
    NeuralNet.prototype.dealTVal = function (tvals) {
        var size = [tvals.rows, tvals.cols];
        var elements = tvals.elements;
        return { size: size, elements: elements };
    };
    NeuralNet.prototype.dealInput = function (input) {
        var size = [input.rows, input.cols];
        var elements = input.elements;
        return { size: size, elements: elements };
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
