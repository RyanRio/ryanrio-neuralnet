import { Matrix, matrix } from '@ryanrio/matrix';
import * as fs from 'fs';

export interface PrePreppedNet {
    weights: Matrix;
    targetValues: Matrix;
    inputLayer: Matrix;
    hiddenLayers: Matrix[];
}
/**
 * @param weights The weights that should be changed
 * @param targetValues The targets that should be changed
 * @param inputLayer the inputLayer values that should be changed
 * @param hiddenLayers the hiddenLayer(s) values that should be changed
 */

class OptionalNet {
    weights?: Matrix;
    targetValues?: Matrix;
    inputLayer?: Matrix;
    hiddenLayers?: Matrix[];

    constructor(weights?: number[][], tVals?: number[], inputs?: number[], hiddenLayers?: number[]) {
        if (weights) {
            this.weights = matrix(weights);
        }
        if (tVals) {
            this.targetValues = matrix([tVals]);
        }
        if (inputs) {
            this.inputLayer = matrix([inputs]);
        }
        if (hiddenLayers) {
            this.hiddenLayers = [matrix([hiddenLayers])];
        }
    }
}
export interface PreppedNet {
    weights: Tensor;
    targets: Tensor;
    inputs: Tensor;
    hiddenLayers: Tensor[];

}
export interface Tensor {
    size: number[];
    elements: number[][];
}

export interface NeuralNetInterface {
    JSON: string;
    adjustNet(optionalNet: OptionalNet): void;
}

interface NetNodeInterface {
    /**
     * notation - l
     */
    layer: number;
    /**
     * notation - k
     */
    fromNode?: number;
    /**
     * notation - j
     */
    toNode?: number;
    /**
     * notation - Value
     */
    Value: number;
}

class NetNode {
    Value: number;

    constructor(Value: number) {
        this.Value = Value;
    }
}

interface LayerInterface {
    /**
     * Which layer we are on
     */
    layer: number;
    /**
     * The nodes in the layer
     */
    xNodes: NetNode[];
    /**
     * The nodes in the layer
     */
    yNodes: NetNode[];
    /**
     * number of nodes in the layer (length of "Nodes")
     */
    Sl: number;
}

class Layer implements LayerInterface {
    /**
     * Which layer we are on
     */
    layer: number;
    /**
     * The nodes in the layer
     */
    xNodes: NetNode[];
    /**
     * number of nodes in the layer (length of "Nodes")
     */
    yNodes: NetNode[];
    /**
     * number of nodes in the layer (length of "Nodes")
     */
    Sl: number;

    constructor(layer: number, xNodes: NetNode[], yNodes: NetNode[], Sl: number) {
        this.layer = layer;
        this.xNodes = xNodes;
        this.Sl = Sl;
        this.yNodes = yNodes;
    }
}

interface NetSpecInterface {
    n_inputNodes: number;
    n_hiddenNodes: number;
    n_outputNodes: number;
}

export class NetSpec implements NetSpecInterface {
    n_inputNodes: number;
    n_hiddenNodes: number;
    n_outputNodes: number;

    constructor(input: number, hidden: number, output: number) {
        this.n_inputNodes = input;
        this.n_hiddenNodes = hidden;
        this.n_outputNodes = output;
    }
}
export class NeuralNet implements NeuralNetInterface {
    private prePreppedNet: PrePreppedNet;
    private preppedNet: PreppedNet;
    JSON: string;
    /**
     * Creates a NeuralNet which has a JSON representation
     * @param net
     */
    constructor(net: PrePreppedNet | NetSpecInterface) {
        if (this.isPrePreppedNet(net)) {
            console.log("PrePreppedNet");
            this.prePreppedNet = net;
            this.preppedNet = this.prepJSON(this.prePreppedNet);
            this.JSON = JSON.stringify(this.preppedNet);
        }
        else {
            const genArray = function (iterator: number): number[] {
                let array = [];
                for (let i = 0; i < iterator; i++) {
                    array.push(Math.random());
                }
                return array;
            };
            const weights: Matrix = matrix([[]]);
            for (let i = 0; i < net.n_hiddenNodes; i++) {
                if (i === 0) {
                    weights.elements[0] = genArray(net.n_inputNodes + 1);
                }
                else {
                    weights.elements.push(genArray(net.n_inputNodes + 1));
                }
            }
            for (let i = 0; i < net.n_hiddenNodes; i++) {
                weights.elements.push(genArray(net.n_outputNodes));
            }
            const inputs: Matrix = matrix([[]]);
            let e = [];
            for (let i = 0; i < net.n_inputNodes + 1; i++) {
                e.push(1.0);
            }
            inputs.elements[0] = e;
            const outputs: Matrix = matrix([[]]);
            outputs.elements[0] = e;
            this.prePreppedNet = {
                inputLayer: matrix(inputs.elements),
                weights: matrix(weights.elements),
                targetValues: matrix(outputs.elements),
                hiddenLayers: [matrix([[]])]
            };
            console.log(this.prePreppedNet.weights.rows);
            this.preppedNet = this.prepJSON(this.prePreppedNet);
            this.JSON = JSON.stringify(this.preppedNet);
        }
        this.initiateNet();
    }

    private isPrePreppedNet(object: any): object is PrePreppedNet {
        return 'member' in object;
    }

    private initiateNet() {
        let net: PreppedNet = this.preppedNet;

        // input
        let xNodes =  this.constructNodes(net.inputs);
        let inputLayerNum = 0;
        let inputSl = net.inputs.elements[0].length;
        let yNodes = this.activate(xNodes);
        let inputLayer = new Layer(inputLayerNum, xNodes, yNodes, inputSl);

        // Hidden Layer
        // xNodes of hidden layer, for now assume there is only one hidden layer
        let xNodesH = this.sumFromWeights(net.weights.elements[0], yNodes);
        let hiddenLayerNum = 1;
        let hiddenSl = net.hiddenLayers[0].elements[0].length;
        let yNodesH = this.activate(xNodesH);
        let hiddenLayer = new Layer(hiddenLayerNum, xNodesH, yNodesH, hiddenSl);

        // Output Layer
        let xNodesO = this.sumFromWeights(net.weights.elements[1], yNodesH);
        let outputLayerNum = 2;
        let outputSl = net.targets.elements[0].length;
        let yNodesO = this.activate(xNodesO);
        let outputLayer = new Layer(outputLayerNum, xNodesH, yNodesH, hiddenSl);


        const inputElements = [];
        for (const node of yNodes) {
            inputElements.push(node.Value);
        }

        const hiddenElements = [];
        for (const node of yNodesH) {
            hiddenElements.push(node.Value);
        }

        const outputElements = [];
        for (const node of yNodesO) {
            outputElements.push(node.Value);
        }


        this.adjustNet(new OptionalNet(undefined, outputElements, inputElements, hiddenElements));
    }

    private sumFromWeights(weights: number[], yNodes: NetNode[]): NetNode[] {
        let returnArray: NetNode[] = [];
        for (let k = 0; k < yNodes.length; k++) {
            let sum = 0;
            for (let weight of weights) {
                sum = weight * yNodes[k].Value;
            }
            returnArray.push(new NetNode(sum));
        }
        return returnArray;
    }

    private activate(activatable: NetNode[]): NetNode[] {
        let returnArray: NetNode[] = [];
        for (let node of activatable) {
            returnArray.push(new NetNode((1 / (1 + Math.exp(-node.Value)))));
        }
        return returnArray;
    }

    private constructNodes(input: Tensor): NetNode[] {
        let returnArray: NetNode[] = [];
        for (const element of input.elements[0]) {
            returnArray.push(new NetNode(element));
        }
        return returnArray;
    }
    /**
     * Changes the prePreppedNet and then update the PreppedNet and the JSON
     * @param optionalNet Object which contains what you want to change about the prePreppedNet
     */
    public adjustNet(optionalNet: OptionalNet) {
        const weights = optionalNet.weights;
        const tVals = optionalNet.targetValues;
        const inputs = optionalNet.inputLayer;
        const hiddenLayers = optionalNet.hiddenLayers;

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
    }

    /**
     * Updates the preppedNet and JSON using the current prePreppedNet
     */
    private update(): void {
        this.preppedNet = this.prepJSON(this.prePreppedNet);
        this.JSON = JSON.stringify(this.preppedNet);
    }
    private dealHidden(layers: Matrix[]): Tensor[] {
        const tensors = [];
        for (const layer of layers) {
            const size = [layer.rows, layer.cols];
            const elements = layer.elements;
            tensors.push({ size, elements });
        }
        return tensors;
    }
    private dealWeights(weights: Matrix): Tensor {
        const size = [weights.rows, weights.cols];
        const elements = weights.elements;
        return { size, elements };
    }
    private dealTVal(tvals: Matrix): Tensor {
        const size = [tvals.rows, tvals.cols];
        const elements = tvals.elements;
        return { size, elements };
    }
    private dealInput(input: Matrix): Tensor {
        const size = [input.rows, input.cols];
        const elements = input.elements;
        return { size, elements };
    }

    /**
     * Change the prepreppednet to a JSON-stringify compatible format
     * @param net
     */
    private prepJSON(net: PrePreppedNet): PreppedNet {
        const weightsJSON = this.dealWeights(net.weights);
        const targetValuesJSON = this.dealTVal(net.targetValues);
        const inputLayerJSON = this.dealInput(net.inputLayer);
        const hiddenLayersJSON = this.dealHidden(net.hiddenLayers);

        const weights = weightsJSON;
        const targets = targetValuesJSON;
        const inputs = inputLayerJSON;
        const hiddenLayers = hiddenLayersJSON;

        return {
            weights,
            targets,
            inputs,
            hiddenLayers
        };

    }
}


