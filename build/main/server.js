"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var matrix_1 = require("@ryanrio/matrix");
var neural_net_1 = require("./lib/neural-net");
var NetFileManager_1 = require("./lib/NetFileManager");
function test() {
    var weights = matrix_1.matrix([[.1, .2], [3, 2]]);
    var targetValues = matrix_1.matrix([[.5, .2]]);
    var inputLayer = matrix_1.matrix([[1, 2]]);
    var hiddenLayers = [matrix_1.matrix([[5, 6]]), matrix_1.matrix([[19, 12]])];
    var net = {
        weights: weights,
        targetValues: targetValues,
        inputLayer: inputLayer,
        hiddenLayers: hiddenLayers
    };
    var myNet = new neural_net_1.NeuralNet(net);
    var myManager = new NetFileManager_1.NetFileManager(myNet);
    return myManager;
}
var manager;
manager = test();
setTimeout(function () {
    var managedNet = manager.net;
    var weightsChange = matrix_1.matrix([[0, .2], [3, 2]]);
    console.log("updating net");
    managedNet.adjustNet({ weights: weightsChange });
    manager.save();
}, 3000);
