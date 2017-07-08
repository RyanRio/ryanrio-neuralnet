import { Matrix } from '@ryanrio/matrix';
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
export interface OptionalNetInterface {
    weights?: Matrix;
    targetValues?: Matrix;
    inputLayer?: Matrix;
    hiddenLayers?: Matrix[];
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
    adjustNet(optionalNet: OptionalNetInterface): void;
}
export declare class NeuralNet implements NeuralNetInterface {
    private prePreppedNet;
    private preppedNet;
    JSON: string;
    /**
     * Creates a NeuralNet which has a JSON representation
     * @param net
     */
    constructor(net: PrePreppedNet);
    /**
     * Changes the prePreppedNet and then update the PreppedNet and the JSON
     * @param optionalNet Object which contains what you want to change about the prePreppedNet
     */
    adjustNet(optionalNet: OptionalNetInterface): void;
    /**
     * Updates the preppedNet and JSON using the current prePreppedNet
     */
    private update();
    private dealHidden(layers);
    private dealWeights(weights);
    private dealTVal(tvals);
    private dealInput(input);
    private getElements(matrix);
    /**
     * Change the prepreppednet to a JSON-stringify compatible format
     * @param net
     */
    private prepJSON(net);
}
