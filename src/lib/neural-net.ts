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

export class NeuralNet implements NeuralNetInterface {
    private prePreppedNet: PrePreppedNet;
    private preppedNet: PreppedNet;
    JSON: string;
    /**
     * Creates a NeuralNet which has a JSON representation
     * @param net
     */
    constructor(net: PrePreppedNet) {
        this.prePreppedNet = net;
        this.preppedNet = this.prepJSON(this.prePreppedNet);
        this.JSON = JSON.stringify(this.preppedNet);
    }

    /**
     * Changes the prePreppedNet and then update the PreppedNet and the JSON
     * @param optionalNet Object which contains what you want to change about the prePreppedNet
     */
    public adjustNet(optionalNet: OptionalNetInterface) {
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
            const elements = this.getElements(layer);
            tensors.push({ size, elements });
        }
        return tensors;
    }
    private dealWeights(weights: Matrix): Tensor {
        const size = [weights.rows, weights.cols];
        const elements = this.getElements(weights);
        return { size, elements };
    }
    private dealTVal(tvals: Matrix): Tensor {
        const size = [tvals.rows, tvals.cols];
        const elements = this.getElements(tvals);
        return { size, elements };
    }
    private dealInput(input: Matrix): Tensor {
        const size = [input.rows, input.cols];
        const elements = this.getElements(input);
        return { size, elements };
    }
    private getElements(matrix: Matrix): number[][] {
        let elements: number[][] = [];
        for (let i = 0; i < matrix.rows; i++) {
            elements.push([]);
            for (let j = 0; j < matrix.cols; j++) {
                elements[i].push(matrix.getElement(i + 1, j + 1));
            }
        }
        return elements;
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


