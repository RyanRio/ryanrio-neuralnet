/**
 * Always listed from top to bottom, ie [1,2,3] is equivalent to
 * [1
 * 2
 * 3]
 */
export declare type Vector = number[];
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
export declare class Matrix implements MatrixI {
    postitionedVectors: AugmentedVector[];
    next(inVector: Vector): void;
    constructor(inVector?: Vector);
    readonly vectors: number[][];
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
export declare type WeightVector = Weight[];
/**
 * Thus a single theta is an array of arrays of weights
 */
export declare type WeightLayer = WeightVector[];
export declare type theta = WeightLayer[];
export declare class NeuralNet {
    _weights: theta;
    weightMatrices: Matrix[];
    numInputs: number;
    numOutcomes: number;
    /**
     * learning rate
     */
    lr: number;
    constructor(inputElements: number, hiddenLayers: number, numOutcomes: number, learningRate: number);
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
    weightVal(layer: number, kVal: number, jVal: number): Weight;
    /**
     *
     * @param filename Desired name of the JSON file, ie "net.json"
     * @param path Desired location of the file, ie "./output/"
     */
    toJSON(filename: string, path: string, data: string): void;
}
