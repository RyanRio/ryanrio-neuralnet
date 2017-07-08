import { matrix, Matrix } from '@ryanrio/matrix';
import { NeuralNet } from './lib/neural-net';
import { NetFileManager } from './lib/NetFileManager';

function test(): NetFileManager {
    let weights: Matrix = matrix([[.1, .2], [3, 2]]);
    let targetValues: Matrix = matrix([[.5, .2]]);
    let inputLayer: Matrix = matrix([[1, 2]]);
    let hiddenLayers: Matrix[] = [matrix([[5, 6]]), matrix([[19, 12]])];

    const net = {
        weights,
        targetValues,
        inputLayer,
        hiddenLayers
    };

    const myNet = new NeuralNet(net);
    const myManager = new NetFileManager(myNet);
    return myManager;
}
let manager: NetFileManager;
manager = test();


setTimeout(() => {
    const managedNet = manager.net;
    const weightsChange = matrix([[0, .2], [3, 2]]);
    console.log("updating net");
    managedNet.adjustNet({ weights: weightsChange });
    manager.save();
}, 3000);

