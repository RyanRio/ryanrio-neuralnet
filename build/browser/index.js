(function (global, factory) {
	typeof exports === 'object' && typeof module !== 'undefined' ? factory(exports, require('fs')) :
	typeof define === 'function' && define.amd ? define(['exports', 'fs'], factory) :
	(factory((global.tslibBase = global.tslibBase || {}),global.fs));
}(this, (function (exports,fs) { 'use strict';

var ManagerLogger = (function () {
    function ManagerLogger(net) {
        this.timesSaved = 1;
        this.net = net;
    }
    ManagerLogger.prototype.log = function (logMessage) {
        this.timesSaved++;
        fs.appendFile('logs/log.txt', logMessage + "\n", function (err) {
            if (err) {
                throw err;
            }
        });
    };
    return ManagerLogger;
}());

var NetFileManager = (function () {
    /**
     * Upon contstruction the net is saved
     * @param net;
     */
    function NetFileManager(net) {
        this.net = net;
        this.logger = new ManagerLogger(this.net);
        this.save();
    }
    NetFileManager.prototype.save = function () {
        var _this = this;
        this.logger.log("Net was just saved: " + this.logger.timesSaved);
        fs.exists("NetJSON/net.json", (function (exists$$1) {
            if (exists$$1) {
                fs.unlink("NetJSON/net.json", function (err) {
                    if (err) {
                        throw err;
                    }
                    else {
                        console.log("File exists, deleting and rewriting");
                        fs.writeFile("NetJSON/net.json", _this.net.JSON, function (err) {
                            if (err) {
                                throw err;
                            }
                        });
                    }
                });
            }
            else {
                console.log("File doesn't exist, writing");
                fs.writeFile("NetJSON/net.json", _this.net.JSON, function (err) {
                    if (err) {
                        throw err;
                    }
                });
            }
        }));
    };
    return NetFileManager;
}());

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

exports.ManagerLogger = ManagerLogger;
exports.NetFileManager = NetFileManager;
exports.NeuralNet = NeuralNet;

Object.defineProperty(exports, '__esModule', { value: true });

})));
//# sourceMappingURL=index.js.map
