"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var fs = require("fs");
var davinci_csv_1 = require("davinci-csv");
var assert = require("assert");
var net_constants_1 = require("./net-constants");
var net_matrix_1 = require("./net-matrix");
var NeuralNet = (function () {
    function NeuralNet(inputElements, hiddenLayers, numOutcomes, learningRate, activateFuncSngl) {
        if (activateFuncSngl === void 0) { activateFuncSngl = net_constants_1.activateFuncDefault; }
        /**
         * The attributeToNumberMap is indexable twice, the first time by name of the attribute to retrieve the last of Fields that are possible
         * in a single attribute. For example Map[safety] which would retrieve a list of MapValues which are in themselves
         * indexable, such as: [high: 0.234, med: 0.111, low: 0.912]. To retrieve the random readonly value simply do
         * Map[safety][high] for example.
         */
        this.attributeToNumberMap = [];
        this.activateFunc = function (inputs) {
            var activatedInputs = [];
            for (var _i = 0, inputs_1 = inputs; _i < inputs_1.length; _i++) {
                var input = inputs_1[_i];
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
        var theta = [];
        for (var i = 1; i <= hiddenLayers; i++) {
            var layer_1 = i;
            /**
             * assembling a WeightLayer
             */
            var weightLayer_1 = Object.create([]);
            // + 2 because +1 to account for indexing and another +1 to account for biased node
            for (var k = 1; k < inputElements + 2; k++) {
                /**
                 * assembling a WeightVector
                 */
                var weightVector = Object.create([]);
                for (var j = 1; j < inputElements + 1; j++) {
                    var weightVal = Math.random();
                    weightVector.push({ value: weightVal, jVal: j, kVal: k, layer: layer_1 });
                }
                weightLayer_1.push(weightVector);
            }
            theta.push(weightLayer_1);
        }
        /**
         * now need to add the final layer of weights which are in-between the last hidden layer and the output layer
         * weight layer number is going to be the number of hidden layers + 1
         */
        var layer = hiddenLayers + 1;
        var weightLayer = Object.create([]);
        // + 2 because +1 to account for indexing and another +1 to account for biased node
        for (var k = 1; k < inputElements + 2; k++) {
            /**
             * assembling a WeightVector
             */
            var weightVector = Object.create([]);
            for (var j = 1; j < numOutcomes + 1; j++) {
                var weightVal = Math.random();
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
        for (var i = 0; i < this._weights.length; i++) {
            var currentLayer = this._weights[i];
            var matrix = new net_matrix_1.Matrix();
            for (var _i = 0, currentLayer_1 = currentLayer; _i < currentLayer_1.length; _i++) {
                var weightArray = currentLayer_1[_i];
                var vector = [];
                for (var _a = 0, weightArray_1 = weightArray; _a < weightArray_1.length; _a++) {
                    var weight = weightArray_1[_a];
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
    NeuralNet.prototype.weightVal = function (layer, kVal, jVal) {
        var weight = this._weights[layer - 1][kVal - 1][jVal - 2];
        return weight;
    };
    /**
     *
     * @param filename Desired name of the JSON file, ie "net.json"
     * @param path Desired location of the file, ie "./output/"
     */
    NeuralNet.prototype.toJSON = function (filename, path, data) {
        var fullPath = path + filename;
        if (fs.existsSync(fullPath)) {
            fs.unlinkSync(fullPath);
            fs.writeFileSync(fullPath, JSON.stringify(data));
        }
    };
    NeuralNet.prototype.getTargetSerial = function (name) {
        var targets = this.targetPossibilities;
        for (var _i = 0, targets_1 = targets; _i < targets_1.length; _i++) {
            var target = targets_1[_i];
            if (target.name === name) {
                return target.serialization;
            }
        }
        throw new Error("target does not exist");
    };
    NeuralNet.prototype.getTargetName = function (serialization) {
        var targets = this.targetPossibilities;
        for (var _i = 0, targets_2 = targets; _i < targets_2.length; _i++) {
            var target = targets_2[_i];
            if (target.serialization === serialization) {
                return target.name;
            }
        }
        throw new Error("target does not exist");
    };
    NeuralNet.prototype.initializeTargets = function (input) {
        var targets = [];
        function createTargetVector(length, oneLocation) {
            var m = [];
            for (var i = 0; i < length; i++) {
                if (i === oneLocation) {
                    m.push(1);
                }
                else {
                    m.push(0);
                }
            }
            return m;
        }
        for (var index = 0; index < input.length; index++) {
            var n = input[index];
            if (typeof n === "string" || typeof n === "number") {
                targets.push({ name: n, serialization: createTargetVector(input.length, index) });
            }
        }
        return targets;
    };
    NeuralNet.prototype.createValues = function (trainingVal) {
        var numberedField = [];
        for (var i = 0; i < trainingVal.length - 1; i++) {
            var attributeName = this.attributeValues[i].name;
            var val = trainingVal[i];
            if (typeof val === 'string') {
                var numVal = this.attributeToNumberMap[attributeName][val];
                numberedField.push(numVal);
            }
            else if (typeof val === 'number') {
                var numVal = val;
                numberedField.push(numVal);
            }
            else {
                throw new Error("value is null, no null allowed");
            }
        }
        return numberedField;
    };
    /**
     * Note that at the end of a csv file a string is required, thus the neural net will look for "end" and not use it to train
     * @param csvFilename relative path of the csv file you want to parse
     * @param dialect options to parse the csv file differently
     * @param attributeMapping the neural net will follow the description of the attributeMapping and targetPossibilities
     * @param targetPossibilities the neural net will follow the description of the attributeMapping and targetPossibilities
     * @param ensureValidity if true check to make sure that the data given by the csv file matches what is described by the attribute mapping and targetPossibilities, defaults to true
     */
    NeuralNet.prototype.TrainWithCSV = function (csvFilename, attributeMapping, targetPossibilities, dialect, ensureValidity) {
        if (ensureValidity === void 0) { ensureValidity = true; }
        this.attributeValues = attributeMapping;
        this.targetPossibilities = this.initializeTargets(targetPossibilities);
        var csv = "";
        var result = fs.readFileSync(csvFilename);
        csv += result;
        var parsedCSV;
        try {
            parsedCSV = davinci_csv_1.parse(csv);
        }
        catch (err) {
            throw err;
        }
        // assign values for all the attributes which ARE PERMANENT after assignmnet
        for (var _i = 0, attributeMapping_1 = attributeMapping; _i < attributeMapping_1.length; _i++) {
            var attribute = attributeMapping_1[_i];
            var currentAttributeToNumberMap = [];
            for (var _a = 0, _b = attribute.values; _a < _b.length; _a++) {
                var value = _b[_a];
                if (typeof value === 'string') {
                    currentAttributeToNumberMap[value] = Math.random();
                }
                else if (typeof value === 'number') {
                    currentAttributeToNumberMap[value.toString()] = value;
                }
            }
            this.attributeToNumberMap[attribute.name] = currentAttributeToNumberMap;
        }
        var endElement = parsedCSV.pop();
        if (endElement) {
            assert.deepEqual(endElement, ["end"], "Last element should be 'end'");
        }
        else {
            throw new Error("CSV File is empty");
        }
        if (ensureValidity) {
            // Check each Line (Array of Fields) for validity
            for (var _c = 0, parsedCSV_1 = parsedCSV; _c < parsedCSV_1.length; _c++) {
                var trainingVal = parsedCSV_1[_c];
                // First ensure that the target is valid
                var target = trainingVal[trainingVal.length - 1];
                if (targetPossibilities.indexOf(target) === -1) {
                    throw new Error("Does not match the target mapping... " + target + " could not be located in " + targetPossibilities);
                }
                // Next check that the other fields values all match the attribute mapping
                // Index is important because it corresponds to the attribute that should be chosen
                for (var index = 0; index < trainingVal.length - 1; index++) {
                    var attributeToCompare = attributeMapping[index];
                    if (!attributeToCompare) {
                        throw new Error("Attribute list does not contain enough values... Length should be " + (trainingVal.length - 1) + ", but it is " + attributeMapping.length);
                    }
                    var value = trainingVal[index];
                    if (attributeToCompare.values.indexOf(value) === -1) {
                        throw new Error("Does not match attribute mapping... " + value + " could not be located in " + attributeToCompare.values);
                    }
                }
            }
        }
        this.y = [];
        /**
         * A training value is a list of fields as defined by the attributeMapping with the target at the end
         * Assumes that validity has been checked
         */
        for (var _d = 0, parsedCSV_2 = parsedCSV; _d < parsedCSV_2.length; _d++) {
            var trainingVal = parsedCSV_2[_d];
            fs.writeFileSync("weights1.txt", JSON.stringify(this.weightMatrices));
            var target = trainingVal[trainingVal.length - 1];
            // make sure target isn't null
            if (typeof target !== "string" && typeof target !== "number") {
                throw new Error("target is invalid");
            }
            var inputs = this.createValues(trainingVal);
            // activate inputs
            var activatedInputs = this.activateFunc(inputs);
            // add bias node to activatedInputs
            activatedInputs = [1].concat(activatedInputs);
            this.y.push(activatedInputs);
            // For example if there is one hidden layer then only propagate to get those values
            for (var i = 0; i < this.hiddenLayers; i++) {
                /**
                 * Check that the matrices will multiply correctly - the number of columns (vectors) of the weights matrix
                 * should be equal to the number of rows (length) of the activatedInputs array
                 *
                 * Note: The bias node exists which has a value of one, so it is added to the activatedInputs every time
                 * and has a value of 1
                 */
                if (activatedInputs.length === this.weightMatrices[i].vectors.length) {
                    var z_1 = [];
                    // multiplication of matrices - take each row of the weights matrix and dot it by the activated inputs
                    var weightMatrixTransposed = this.weightMatrices[i].toRowFormat();
                    for (var _e = 0, weightMatrixTransposed_1 = weightMatrixTransposed; _e < weightMatrixTransposed_1.length; _e++) {
                        var row = weightMatrixTransposed_1[_e];
                        z_1.push(net_matrix_1.dotProduct(row, activatedInputs));
                    }
                    // Make new activated inputs to either prep for next layer or prep to create output values
                    inputs = z_1;
                    activatedInputs = this.activateFunc(inputs);
                    activatedInputs = [1].concat(activatedInputs);
                    this.y.push(activatedInputs);
                }
            }
            var lastWeightMatrix = this.weightMatrices[this.weightMatrices.length - 1];
            var lastWeightMatrixTransposed = lastWeightMatrix.toRowFormat();
            var z = [];
            for (var _f = 0, lastWeightMatrixTransposed_1 = lastWeightMatrixTransposed; _f < lastWeightMatrixTransposed_1.length; _f++) {
                var row = lastWeightMatrixTransposed_1[_f];
                z.push(net_matrix_1.dotProduct(row, activatedInputs));
            }
            var outputs = z;
            var activatedOutputs = this.activateFunc(outputs);
            this.y.push(activatedOutputs);
            // evaluate cost
            var sum = 0;
            for (var j = 0; j < activatedOutputs.length; j++) {
                var t = this.getTargetSerial(target)[j];
                this.t = this.getTargetSerial(target);
                var y = activatedOutputs[j];
                var val = Math.pow((t - y), 2);
                console.log("t: " + t + ", y: " + y + ", t-y: " + (t - y) + ", (t-y)^2: " + Math.pow((t - y), 2));
                sum = sum + val;
            }
            var cost = 0.5 * sum;
            console.log("cost: " + cost);
            fs.writeFileSync("weights2.txt", JSON.stringify(this.weightMatrices));
            // back-propagation
            for (var l = 0; l < this.hiddenLayers + 1; l++) {
                // k is from
                // console.log(l);
                // console.log(this.weightMatrices.length);
                for (var k = 0; k < this.weightMatrices[l].vectors.length; k++) {
                    for (var j = 0; j < this.weightMatrices[l].vectors[k].length; j++) {
                        var weight = this.weightMatrices[l].vectors[k][j];
                        var p1 = ((-1) * this.lambda(j, l + 1));
                        // console.log("p1: " + p1);
                        var p2 = this.y[l][k];
                        // console.log("p2: " + p2);
                        this.weightMatrices[l].vectors[k][j] = weight + (p1 * p2);
                    }
                }
            }
        }
        return this.weightMatrices;
    };
    NeuralNet.prototype.lambda = function (j, l) {
        // if were at the output layer return a number
        if (l === this.y.length - 1) {
            var t = this.t;
            var ysubl = this.y[l - 1];
            // console.log(ysubl);
            var ysubL = this.y[l];
            // console.log(ysubL);
            console.log(t.length);
            console.log(t[j] + "j: " + j);
            var returnval = ((t[j] - ysubl[j]) * ysubL[j] * (1 - ysubL[j]));
            // console.log("lambda return: " + returnval);
            return returnval;
        }
        else {
            var sum = 0;
            // find sum... recurse if neccessary
            for (var k = 1; k < this.y[l - 1].length; k++) {
                var p1 = this.lambda(k, l + 1);
                // console.log("p1: " + p1);
                var p2 = this.getWeight(k, j, l - 1);
                // console.log("p2: " + p2);
                sum = sum + (p1 * p2);
            }
            // console.log("sum: " + sum);
            return (sum * this.y[l][j] * (1 - this.y[l][j]));
        }
    };
    NeuralNet.prototype.getWeight = function (j, k, l) {
        return this.weightMatrices[l].vectors[k][j];
    };
    NeuralNet.prototype.getAttributeMapping = function () {
        var returnArr = [];
        var aValues = this.attributeValues;
        var Mapping = this.attributeToNumberMap;
        for (var _i = 0, aValues_1 = aValues; _i < aValues_1.length; _i++) {
            var aVal = aValues_1[_i];
            for (var _a = 0, _b = aVal.values; _a < _b.length; _a++) {
                var val = _b[_a];
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
    };
    return NeuralNet;
}());
exports.NeuralNet = NeuralNet;
