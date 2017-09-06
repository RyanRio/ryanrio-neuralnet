import * as fs from 'fs';
import { parse, Dialect, Field } from 'davinci-csv';
import * as assert from 'assert';
import { activateFuncDefault } from './net-constants';
import { AugmentedVector, Matrix, MatrixI, Vector, dotProduct } from './net-matrix';

interface Attribute {
    name: string;
    values: (string | number | null)[];
}

interface Target {
    name: string | number;
    serialization: number[];
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
interface MapVal {
    [key: string]: number;
}
interface MapValRaised {
    [key: string]: MapVal[];
}
export class NeuralNet {
    public activateFunc: (inputs: number[]) => number[];
    public _weights: theta;
    public weightMatrices: Matrix[];
    /**
     * numInputs should be the number of values that are going to be entered into a neuralNet (as a trainingValue)
     * Note that this is NOT the number of possible inputs
     */
    numInputs: number;
    /**
     * numOutcomes should be the number of possible outcomes
     */
    numOutcomes: number;
    /**
     * essentially the number of layers to propagate through
     */
    hiddenLayers: number;
    /**
     * The attributeToNumberMap is indexable twice, the first time by name of the attribute to retrieve the last of Fields that are possible
     * in a single attribute. For example Map[safety] which would retrieve a list of MapValues which are in themselves
     * indexable, such as: [high: 0.234, med: 0.111, low: 0.912]. To retrieve the random readonly value simply do
     * Map[safety][high] for example.
     */
    attributeToNumberMap: MapValRaised[] = [];
    /**
     * Possible attributes that the neural net can have, each attribute has a name and a list of values
     * For example, {name: safety, values: high, med, low}
     */
    attributeValues: Attribute[];
    /**
     * Mapping of an array to a string
     * For example if the possible outcomes are "1" and "2" targetPossibilities would be
     * {
     *  [1, 0]: "1",
     *  [0, 1]: "2"
     * }
     */
    targetPossibilities: Target[];
    /**
     * learning rate
     */
    lr: number;
    /**
     * An array containing the multiple arrays of y's (activated values). Necessary for back-propagation
     */
    private y: number[][];
    /**
     * An array containing the values of the targets (serials). Necessary for back-propagation
     */
    private t: number[];
    constructor(inputElements: number, hiddenLayers: number, numOutcomes: number, learningRate: number, activateFuncSngl = activateFuncDefault) {
        this.activateFunc = (inputs: number[]): number[] => {
            let activatedInputs: number[] = [];
            for (let input of inputs) {
                activatedInputs.push(activateFuncSngl(input));

            }
            return activatedInputs;
        };
        this.hiddenLayers = hiddenLayers;
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
    public toJSON(filename: string, path: string, data: any) {
        const fullPath = path + filename;
        if (fs.existsSync(fullPath)) {
            fs.unlinkSync(fullPath);
            fs.writeFileSync(fullPath, JSON.stringify(data));
        }
    }

    getTargetSerial(name: string | number): number[] {
        const targets = this.targetPossibilities;
        for (let target of targets) {
            if (target.name === name) {
                return target.serialization;
            }
        }
        throw new Error("target does not exist");
    }

    getTargetName(serialization: number[]): string | number | undefined {
        const targets = this.targetPossibilities;
        for (let target of targets) {
            if (target.serialization === serialization) {
                return target.name;
            }
        }
        throw new Error("target does not exist");
    }

    initializeTargets(input: Field[]): Target[] {
        let targets: Target[] = [];
        function createTargetVector(length: number, oneLocation: number) {
            let m = [];
            for (let i = 0; i < length; i++) {
                if (i === oneLocation) {
                    m.push(1);
                }
                else {
                    m.push(0);
                }
            }

            return m;
        }

        for (let index = 0; index < input.length; index++) {
            let n = input[index];
            if (typeof n === "string" || typeof n === "number") {
                targets.push({ name: n, serialization: createTargetVector(input.length, index) });
            }
        }

        return targets;
    }

    createValues(trainingVal: Field[]): number[] {
        let numberedField: number[] = [];
        for (let i = 0; i < trainingVal.length - 1; i++) {
            let attributeName = this.attributeValues[i].name;
            let val = trainingVal[i];
            if (typeof val === 'string') {
                let numVal: number = this.attributeToNumberMap[attributeName][val];
                numberedField.push(numVal);
            }
            else if (typeof val === 'number') {
                let numVal = val;
                numberedField.push(numVal);
            }
            else {
                throw new Error("value is null, no null allowed");
            }
        }
        return numberedField;
    }

    /**
     * Note that at the end of a csv file a string is required, thus the neural net will look for "end" and not use it to train
     * @param csvFilename relative path of the csv file you want to parse
     * @param dialect options to parse the csv file differently
     * @param attributeMapping the neural net will follow the description of the attributeMapping and targetPossibilities
     * @param targetPossibilities the neural net will follow the description of the attributeMapping and targetPossibilities
     * @param ensureValidity if true check to make sure that the data given by the csv file matches what is described by the attribute mapping and targetPossibilities, defaults to true
     */
    TrainWithCSV(csvFilename: string, attributeMapping: Attribute[], targetPossibilities: Field[], dialect?: Dialect, ensureValidity = true): Matrix[] {
        this.attributeValues = attributeMapping;
        this.targetPossibilities = this.initializeTargets(targetPossibilities);

        let csv = "";
        const result = fs.readFileSync(csvFilename);
        csv += result;
        let parsedCSV: Field[][];
        try {
            parsedCSV = parse(csv);
        }
        catch (err) {
            throw err;
        }

        // assign values for all the attributes which ARE PERMANENT after assignmnet
        for (let attribute of attributeMapping) {
            let currentAttributeToNumberMap: MapVal[] = [];
            for (let value of attribute.values) {
                if (typeof value === 'string') {
                    currentAttributeToNumberMap[value] = Math.random();
                }
                else if (typeof value === 'number') {
                    currentAttributeToNumberMap[value.toString()] = value;
                }
            }
            this.attributeToNumberMap[attribute.name] = currentAttributeToNumberMap;
        }
        let endElement = parsedCSV.pop();
        if (endElement) {
            assert.deepEqual(endElement, ["end"], "Last element should be 'end'");
        }
        else {
            throw new Error("CSV File is empty");
        }

        if (ensureValidity) {
            // Check each Line (Array of Fields) for validity
            for (let trainingVal of parsedCSV) {
                // First ensure that the target is valid
                let target = trainingVal[trainingVal.length - 1];
                if (targetPossibilities.indexOf(<string>target) === -1) {
                    throw new Error(`Does not match the target mapping... ${target} could not be located in ${targetPossibilities}`);
                }
                // Next check that the other fields values all match the attribute mapping
                // Index is important because it corresponds to the attribute that should be chosen
                for (let index = 0; index < trainingVal.length - 1; index++) {
                    const attributeToCompare = attributeMapping[index];
                    if (!attributeToCompare) {
                        throw new Error(`Attribute list does not contain enough values... Length should be ${trainingVal.length - 1}, but it is ${attributeMapping.length}`);
                    }
                    const value = trainingVal[index];
                    if (attributeToCompare.values.indexOf(value) === -1) {
                        throw new Error(`Does not match attribute mapping... ${value} could not be located in ${attributeToCompare.values}`);
                    }
                }
            }
        }

        this.y = [];
        /**
         * A training value is a list of fields as defined by the attributeMapping with the target at the end
         * Assumes that validity has been checked
         */
        for (let trainingVal of parsedCSV) {
            fs.writeFileSync("weights1.txt", JSON.stringify(this.weightMatrices));
            let target = trainingVal[trainingVal.length - 1];
            // make sure target isn't null
            if (typeof target !== "string" && typeof target !== "number") {
                throw new Error("target is invalid");
            }
            let inputs = this.createValues(trainingVal);
            // activate inputs
            let activatedInputs = this.activateFunc(inputs);
            // add bias node to activatedInputs
            activatedInputs = [1].concat(activatedInputs);
            this.y.push(activatedInputs);


            // For example if there is one hidden layer then only propagate to get those values
            for (let i = 0; i < this.hiddenLayers; i++) {
                /**
                 * Check that the matrices will multiply correctly - the number of columns (vectors) of the weights matrix
                 * should be equal to the number of rows (length) of the activatedInputs array
                 *
                 * Note: The bias node exists which has a value of one, so it is added to the activatedInputs every time
                 * and has a value of 1
                 */
                if (activatedInputs.length === this.weightMatrices[i].vectors.length) {
                    let z: number[] = [];
                    // multiplication of matrices - take each row of the weights matrix and dot it by the activated inputs
                    let weightMatrixTransposed = this.weightMatrices[i].toRowFormat();
                    for (let row of weightMatrixTransposed) {
                        z.push(dotProduct(row, activatedInputs));
                    }
                    // Make new activated inputs to either prep for next layer or prep to create output values
                    inputs = z;
                    activatedInputs = this.activateFunc(inputs);
                    activatedInputs = [1].concat(activatedInputs);
                    this.y.push(activatedInputs);
                }
            }
            let lastWeightMatrix = this.weightMatrices[this.weightMatrices.length - 1];
            let lastWeightMatrixTransposed = lastWeightMatrix.toRowFormat();
            let z: number[] = [];
            for (let row of lastWeightMatrixTransposed) {
                z.push(dotProduct(row, activatedInputs));
            }
            let outputs = z;
            let activatedOutputs = this.activateFunc(outputs);
            this.y.push(activatedOutputs);
            // evaluate cost
            let sum = 0;
            for (let j = 0; j < activatedOutputs.length; j++) {
                let t = this.getTargetSerial(target)[j];
                this.t = this.getTargetSerial(target);
                let y = activatedOutputs[j];
                let val = Math.pow((t - y), 2);
                console.log(`t: ${t}, y: ${y}, t-y: ${t - y}, (t-y)^2: ${Math.pow((t - y), 2)}`);
                sum = sum + val;
            }
            let cost = 0.5 * sum;
            console.log("cost: " + cost);
            fs.writeFileSync("weights2.txt", JSON.stringify(this.weightMatrices));
            // back-propagation

            for (let l = 0; l < this.hiddenLayers + 1; l++) {
                // k is from
                // console.log(l);
                // console.log(this.weightMatrices.length);
                for (let k = 0; k < this.weightMatrices[l].vectors.length; k++) {
                    for (let j = 0; j < this.weightMatrices[l].vectors[k].length; j++) {
                        const weight = this.weightMatrices[l].vectors[k][j];
                        let p1 = ((-1) * this.lambda(j, l + 1));
                        // console.log("p1: " + p1);
                        let p2 = this.y[l][k];
                        // console.log("p2: " + p2);
                        this.weightMatrices[l].vectors[k][j] = weight + (p1 * p2);
                    }
                }

            }

        }
        return this.weightMatrices;
    }

    private lambda(j: number, l: number): number {
        // if were at the output layer return a number
        if (l === this.y.length - 1) {
            let t = this.t;
            let ysubl = this.y[l - 1];
            // console.log(ysubl);
            let ysubL = this.y[l];
            // console.log(ysubL);
            console.log(t.length);
            console.log(t[j] + "j: " + j);
            let returnval = ((t[j] - ysubl[j]) * ysubL[j] * (1 - ysubL[j]));
            // console.log("lambda return: " + returnval);
            return returnval;
        }
        else {
            let sum = 0;
            // find sum... recurse if neccessary
            for (let k = 1; k < this.y[l - 1].length; k++) {
                let p1 = this.lambda(k, l + 1);
                // console.log("p1: " + p1);
                let p2 = this.getWeight(k, j, l - 1);
                // console.log("p2: " + p2);
                sum = sum + (p1 * p2);
            }
            // console.log("sum: " + sum);
            return (sum * this.y[l][j] * (1 - this.y[l][j]));
        }

    }

    private getWeight(j: number, k: number, l: number): number {
        return this.weightMatrices[l].vectors[k][j];
    }
    getAttributeMapping(): any[] {
        let returnArr: any[] = [];
        let aValues = this.attributeValues;
        let Mapping = this.attributeToNumberMap;
        for (let aVal of aValues) {
            for (let val of aVal.values) {
                if (typeof val === 'string') {
                    returnArr.push({
                        name: val,
                        value: Mapping[aVal.name][val]
                    });
                }
                else if (typeof val === 'number') {
                    returnArr.push({
                        name: val,
                        value: Mapping[aVal.name][val.toString()]
                    });
                }
            }
        }
        return returnArr;
    }

}


