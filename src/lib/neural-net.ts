import * as fs from 'fs';

/**
 * Always listed from top to bottom, ie [1,2,3] is equivalent to
 * [1
 * 2
 * 3]
 */
export type Vector = number[];

export interface AugmentedVector {
    vector: Vector;
    /**
     * what column number, USING non-zero indexing (first is '1')
     */
    posInMatrix: number;
}
/**
 * Column based 'vector-based'
 * IE the matrix made up of two vectors [1,2,3] and [4,5,6] really looks like
 * 1 4
 * 2 5
 * 3 6
 */
export interface MatrixI {
    postitionedVectors: AugmentedVector[];
    next(vector: Vector): void;
}

export class Matrix implements MatrixI {
    postitionedVectors: AugmentedVector[];
    public next(inVector: Vector) {
        this.postitionedVectors.push(
            { vector: inVector, posInMatrix: this.postitionedVectors.length + 1 }
        );
    }
    constructor(inVector?: Vector) {
        if (inVector) {
            this.postitionedVectors = [{ vector: inVector, posInMatrix: 1 }];
        }
        else {
            this.postitionedVectors = [];
        }
    }
    get vectors() {
        let realVectors: Vector[] = [];
        for (let vector of this.postitionedVectors) {
            realVectors.push(vector.vector);
        }
        return realVectors;
    }
}

export interface Weight {
    value: number;
    /**
     * to Node number
     */
    jVal: number;
    /**
     * from node number
     */
    kVal: number;
    layer: number;
}
export type WeightVector = Weight[];
/**
 * Thus a single theta is an array of arrays of weights
 */
export type WeightLayer = WeightVector[];
export type theta = WeightLayer[];

export class NeuralNet {
    public _weights: theta;
    public weightMatrices: Matrix[];
    numInputs: number;
    numOutcomes: number;
    /**
     * learning rate
     */
    lr: number;
    constructor(inputElements: number, hiddenLayers: number, numOutcomes: number, learningRate: number) {
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
        let theta = [];
        for (let i = 1; i <= hiddenLayers; i++) {
            const layer = i;
            /**
             * assembling a WeightLayer
             */
            let weightLayer: WeightLayer = Object.create([]);
            // + 2 because +1 to account for indexing and another +1 to account for biased node
            for (let k = 1; k < inputElements + 2; k++) {
                /**
                 * assembling a WeightVector
                 */
                let weightVector: WeightVector = Object.create([]);
                for (let j = 1; j < inputElements + 1; j++) {
                    const weightVal = Math.random();
                    weightVector.push({ value: weightVal, jVal: j, kVal: k, layer: layer });
                }
                weightLayer.push(weightVector);
            }
            theta.push(weightLayer);
        }
        /**
         * now need to add the final layer of weights which are in-between the last hidden layer and the output layer
         * weight layer number is going to be the number of hidden layers + 1
         */
        const layer = hiddenLayers + 1;
        let weightLayer: WeightLayer = Object.create([]);
        // + 2 because +1 to account for indexing and another +1 to account for biased node
        for (let k = 1; k < inputElements + 2; k++) {
            /**
             * assembling a WeightVector
             */
            let weightVector: WeightVector = Object.create([]);
            for (let j = 1; j < numOutcomes + 1; j++) {
                const weightVal = Math.random();
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
        for (let i = 0; i < this._weights.length; i++) {
            let currentLayer = this._weights[i];
            let matrix: Matrix = new Matrix();
            for (let weightArray of currentLayer) {
                let vector: Vector = [];
                for (let weight of weightArray) {
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
    public weightVal(layer: number, kVal: number, jVal: number): Weight {
        let weight = this._weights[layer - 1][kVal - 1][jVal - 2];
        return weight;
    }
    /**
     *
     * @param filename Desired name of the JSON file, ie "net.json"
     * @param path Desired location of the file, ie "./output/"
     */
    public toJSON(filename: string, path: string, data: string) {
        const fullPath = path + filename;
        if (fs.existsSync(fullPath)) {
            fs.unlinkSync(fullPath);
            fs.writeFileSync(fullPath, data);
        }
    }
}


