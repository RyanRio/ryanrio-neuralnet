import { matrix } from '@ryanrio/matrix';
import { NeuralNet } from './lib/neural-net';
import { NetFileManager } from './lib/NetFileManager';
function test() {
    var weights = matrix([[.1, .2], [3, 2]]);
    var targetValues = matrix([[.5, .2]]);
    var inputLayer = matrix([[1, 2]]);
    var hiddenLayers = [matrix([[5, 6]]), matrix([[19, 12]])];
    var net = {
        weights: weights,
        targetValues: targetValues,
        inputLayer: inputLayer,
        hiddenLayers: hiddenLayers
    };
    var myNet = new NeuralNet(net);
    var myManager = new NetFileManager(myNet);
    return myManager;
}
var manager;
manager = test();
setTimeout(function () {
    var managedNet = manager.net;
    var weightsChange = matrix([[0, .2], [3, 2]]);
    console.log("updating net");
    managedNet.adjustNet({ weights: weightsChange });
    manager.save();
}, 3000);
