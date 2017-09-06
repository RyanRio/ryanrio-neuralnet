"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var neural_net_1 = require("./neural-net");
var assert = require("assert");
var fs = require("fs");
describe("neural-net(rw)", function () {
    describe("csv parsing", function () {
        var net = new neural_net_1.NeuralNet(6, 1, 4, 1);
        var csv = net.TrainWithCSV("./ReferenceFiles/testCarTest.csv", [
            { name: "buying", values: ["vhigh", "high", "med", "low"] },
            { name: "maint", values: ["vhigh", "high", "med", "low"] },
            { name: "doors", values: [2, 3, 4, "5more"] },
            { name: "persons", values: [2, 4, "more"] },
            { name: "lug_boot", values: ["small", "med", "big"] },
            { name: "safety", values: ["low", "med", "high"] }
        ], ["unacc", "acc", "good", "vgood"]);
        fs.writeFileSync("weightsAfter", JSON.stringify(csv));
        it("target vectors are made correctly", function () {
            var targets = net.targetPossibilities;
            assert.deepEqual(targets[0].serialization, [1, 0, 0, 0]);
            assert.deepEqual(targets[0].name, "unacc");
            assert.deepEqual(targets[1].serialization, [0, 1, 0, 0]);
            assert.deepEqual(targets[1].name, "acc");
            assert.deepEqual(targets[2].serialization, [0, 0, 1, 0]);
            assert.deepEqual(targets[2].name, "good");
            assert.deepEqual(targets[3].serialization, [0, 0, 0, 1]);
            assert.deepEqual(targets[3].name, "vgood");
        });
        it("check attribute mapping", function () {
            fs.writeFileSync("./ReferenceFiles/net_attribute_map.txt", JSON.stringify(net.getAttributeMapping()));
        });
        it("write out preBackProp weights and postBackProp weights", function () {
            var postWeights = net.weightMatrices;
            var postBackPropWeights = [];
            for (var _i = 0, postWeights_1 = postWeights; _i < postWeights_1.length; _i++) {
                var weight = postWeights_1[_i];
                postBackPropWeights.push(weight.vectors);
            }
            fs.writeFileSync("./ReferenceFiles/postBackPropWeights.txt", JSON.stringify(postBackPropWeights));
        });
    });
    describe("weight matrix assembly test", function () {
        var net = new neural_net_1.NeuralNet(2, 1, 2, 1);
        it("should get created correctly", function () {
            assert.deepEqual(net.weightMatrices.length, 2);
        });
        describe("first matrix should be correct", function () {
            var firstMatrix = net.weightMatrices[0].vectors;
            assert.deepEqual(firstMatrix.length, 3);
            for (var _i = 0, firstMatrix_1 = firstMatrix; _i < firstMatrix_1.length; _i++) {
                var vector = firstMatrix_1[_i];
                assert.deepEqual(vector.length, 2);
            }
        });
    });
    describe("weight-getting testing", function () {
        it("should get the correct weight, #1", function () {
            var net = new neural_net_1.NeuralNet(2, 1, 2, 1);
            var weight = net.weightVal(1, 3, 2);
            assert.deepEqual(weight.jVal, 1);
            assert.deepEqual(weight.kVal, 3);
            assert.deepEqual(weight.layer, 1);
        });
        it("should get the correct weight, #2", function () {
            var net = new neural_net_1.NeuralNet(2, 1, 2, 1);
            var weight = net.weightVal(1, 3, 3);
            assert.deepEqual(weight.jVal, 2);
            assert.deepEqual(weight.kVal, 3);
            assert.deepEqual(weight.layer, 1);
        });
    });
    describe("_weights-test with one hidden layer", function () {
        var net = new neural_net_1.NeuralNet(2, 1, 2, 1);
        it("number of weight layers", function () {
            assert.deepEqual(net._weights.length, 2);
        });
        it("going into first layer", function () {
            var firstLayer = net._weights[0];
            assert.deepEqual(firstLayer.length, 3);
        });
        it("examining _weights of first layer", function () {
            var firstLayer = net._weights[0];
            // first ensure that there are the correct number of _weights in each vector of the first layer
            for (var _i = 0, firstLayer_1 = firstLayer; _i < firstLayer_1.length; _i++) {
                var vector = firstLayer_1[_i];
                assert.deepEqual(vector.length, 2);
            }
            // examine first vector
            var firstVector = firstLayer[0]; // biased node
            assert.deepEqual(firstVector[0].jVal, 1);
            assert.deepEqual(firstVector[0].kVal, 1);
            assert.deepEqual(firstVector[0].layer, 1);
            assert.deepEqual(firstVector[1].jVal, 2);
            assert.deepEqual(firstVector[1].kVal, 1);
            assert.deepEqual(firstVector[1].layer, 1);
            // examine second vector
            var secondVector = firstLayer[1];
            assert.deepEqual(secondVector[0].jVal, 1);
            assert.deepEqual(secondVector[0].kVal, 2);
            assert.deepEqual(secondVector[0].layer, 1);
            assert.deepEqual(secondVector[1].jVal, 2);
            assert.deepEqual(secondVector[1].kVal, 2);
            assert.deepEqual(secondVector[1].layer, 1);
            // examine final vector
            var thirdVector = firstLayer[2];
            assert.deepEqual(thirdVector[0].jVal, 1);
            assert.deepEqual(thirdVector[0].kVal, 3);
            assert.deepEqual(thirdVector[0].layer, 1);
            assert.deepEqual(thirdVector[1].jVal, 2);
            assert.deepEqual(thirdVector[1].kVal, 3);
            assert.deepEqual(thirdVector[1].layer, 1);
        });
        it("going into final layer", function () {
            var finalLayer = net._weights[1];
            assert.deepEqual(finalLayer.length, 3);
        });
        it("examining _weights of final layer", function () {
            var finalLayer = net._weights[1];
            // first ensure that there are the correct number of _weights in each vector of the first layer
            for (var _i = 0, finalLayer_1 = finalLayer; _i < finalLayer_1.length; _i++) {
                var vector = finalLayer_1[_i];
                assert.deepEqual(vector.length, 2);
            }
            // examine first vector
            var firstVector = finalLayer[0]; // biased node
            assert.deepEqual(firstVector[0].jVal, 1);
            assert.deepEqual(firstVector[0].kVal, 1);
            assert.deepEqual(firstVector[0].layer, 2);
            assert.deepEqual(firstVector[1].jVal, 2);
            assert.deepEqual(firstVector[1].kVal, 1);
            assert.deepEqual(firstVector[1].layer, 2);
            // examine second vector
            var secondVector = finalLayer[1];
            assert.deepEqual(secondVector[0].jVal, 1);
            assert.deepEqual(secondVector[0].kVal, 2);
            assert.deepEqual(secondVector[0].layer, 2);
            assert.deepEqual(secondVector[1].jVal, 2);
            assert.deepEqual(secondVector[1].kVal, 2);
            assert.deepEqual(secondVector[1].layer, 2);
            // examine final vector
            var thirdVector = finalLayer[2];
            assert.deepEqual(thirdVector[0].jVal, 1);
            assert.deepEqual(thirdVector[0].kVal, 3);
            assert.deepEqual(thirdVector[0].layer, 2);
            assert.deepEqual(thirdVector[1].jVal, 2);
            assert.deepEqual(thirdVector[1].kVal, 3);
            assert.deepEqual(thirdVector[1].layer, 2);
        });
    });
    describe("_weights-test with two layers", function () {
        var net = new neural_net_1.NeuralNet(2, 2, 2, 1);
        it("number of weight layers", function () {
            assert.deepEqual(net._weights.length, 3);
        });
        it("going into first layer", function () {
            var firstLayer = net._weights[0];
            assert.deepEqual(firstLayer.length, 3);
        });
        it("examining _weights of first layer", function () {
            var firstLayer = net._weights[0];
            // first ensure that there are the correct number of _weights in each vector of the first layer
            for (var _i = 0, firstLayer_2 = firstLayer; _i < firstLayer_2.length; _i++) {
                var vector = firstLayer_2[_i];
                assert.deepEqual(vector.length, 2);
            }
            // examine first vector
            var firstVector = firstLayer[0]; // biased node
            assert.deepEqual(firstVector[0].jVal, 1);
            assert.deepEqual(firstVector[0].kVal, 1);
            assert.deepEqual(firstVector[0].layer, 1);
            assert.deepEqual(firstVector[1].jVal, 2);
            assert.deepEqual(firstVector[1].kVal, 1);
            assert.deepEqual(firstVector[1].layer, 1);
            // examine second vector
            var secondVector = firstLayer[1];
            assert.deepEqual(secondVector[0].jVal, 1);
            assert.deepEqual(secondVector[0].kVal, 2);
            assert.deepEqual(secondVector[0].layer, 1);
            assert.deepEqual(secondVector[1].jVal, 2);
            assert.deepEqual(secondVector[1].kVal, 2);
            assert.deepEqual(secondVector[1].layer, 1);
            // examine final vector
            var thirdVector = firstLayer[2];
            assert.deepEqual(thirdVector[0].jVal, 1);
            assert.deepEqual(thirdVector[0].kVal, 3);
            assert.deepEqual(thirdVector[0].layer, 1);
            assert.deepEqual(thirdVector[1].jVal, 2);
            assert.deepEqual(thirdVector[1].kVal, 3);
            assert.deepEqual(thirdVector[1].layer, 1);
        });
        it("going into second layer", function () {
            var secondLayer = net._weights[1];
            assert.deepEqual(secondLayer.length, 3);
        });
        it("examining _weights of second layer", function () {
            var secondLayer = net._weights[1];
            // first ensure that there are the correct number of _weights in each vector of the first layer
            for (var _i = 0, secondLayer_1 = secondLayer; _i < secondLayer_1.length; _i++) {
                var vector = secondLayer_1[_i];
                assert.deepEqual(vector.length, 2);
            }
            // examine first vector
            var firstVector = secondLayer[0]; // biased node
            assert.deepEqual(firstVector[0].jVal, 1);
            assert.deepEqual(firstVector[0].kVal, 1);
            assert.deepEqual(firstVector[0].layer, 2);
            assert.deepEqual(firstVector[1].jVal, 2);
            assert.deepEqual(firstVector[1].kVal, 1);
            assert.deepEqual(firstVector[1].layer, 2);
            // examine second vector
            var secondVector = secondLayer[1];
            assert.deepEqual(secondVector[0].jVal, 1);
            assert.deepEqual(secondVector[0].kVal, 2);
            assert.deepEqual(secondVector[0].layer, 2);
            assert.deepEqual(secondVector[1].jVal, 2);
            assert.deepEqual(secondVector[1].kVal, 2);
            assert.deepEqual(secondVector[1].layer, 2);
            // examine final vector
            var thirdVector = secondLayer[2];
            assert.deepEqual(thirdVector[0].jVal, 1);
            assert.deepEqual(thirdVector[0].kVal, 3);
            assert.deepEqual(thirdVector[0].layer, 2);
            assert.deepEqual(thirdVector[1].jVal, 2);
            assert.deepEqual(thirdVector[1].kVal, 3);
            assert.deepEqual(thirdVector[1].layer, 2);
        });
        it("going into final layer", function () {
            var finalLayer = net._weights[2];
            assert.deepEqual(finalLayer.length, 3);
        });
        it("examining _weights of final layer", function () {
            var finalLayer = net._weights[2];
            // first ensure that there are the correct number of _weights in each vector of the first layer
            for (var _i = 0, finalLayer_2 = finalLayer; _i < finalLayer_2.length; _i++) {
                var vector = finalLayer_2[_i];
                assert.deepEqual(vector.length, 2);
            }
            // examine first vector
            var firstVector = finalLayer[0]; // biased node
            assert.deepEqual(firstVector[0].jVal, 1);
            assert.deepEqual(firstVector[0].kVal, 1);
            assert.deepEqual(firstVector[0].layer, 3);
            assert.deepEqual(firstVector[1].jVal, 2);
            assert.deepEqual(firstVector[1].kVal, 1);
            assert.deepEqual(firstVector[1].layer, 3);
            // examine second vector
            var secondVector = finalLayer[1];
            assert.deepEqual(secondVector[0].jVal, 1);
            assert.deepEqual(secondVector[0].kVal, 2);
            assert.deepEqual(secondVector[0].layer, 3);
            assert.deepEqual(secondVector[1].jVal, 2);
            assert.deepEqual(secondVector[1].kVal, 2);
            assert.deepEqual(secondVector[1].layer, 3);
            // examine final vector
            var thirdVector = finalLayer[2];
            assert.deepEqual(thirdVector[0].jVal, 1);
            assert.deepEqual(thirdVector[0].kVal, 3);
            assert.deepEqual(thirdVector[0].layer, 3);
            assert.deepEqual(thirdVector[1].jVal, 2);
            assert.deepEqual(thirdVector[1].kVal, 3);
            assert.deepEqual(thirdVector[1].layer, 3);
        });
    });
    describe("_weights test with non-equal input elements and output elements", function () {
        var net = new neural_net_1.NeuralNet(2, 2, 1, 1);
        it("number of weight layers", function () {
            assert.deepEqual(net._weights.length, 3);
        });
        it("going into first layer", function () {
            var firstLayer = net._weights[0];
            assert.deepEqual(firstLayer.length, 3);
        });
        it("examining _weights of first layer", function () {
            var firstLayer = net._weights[0];
            // first ensure that there are the correct number of _weights in each vector of the first layer
            for (var _i = 0, firstLayer_3 = firstLayer; _i < firstLayer_3.length; _i++) {
                var vector = firstLayer_3[_i];
                assert.deepEqual(vector.length, 2);
            }
            // examine first vector
            var firstVector = firstLayer[0]; // biased node
            assert.deepEqual(firstVector[0].jVal, 1);
            assert.deepEqual(firstVector[0].kVal, 1);
            assert.deepEqual(firstVector[0].layer, 1);
            assert.deepEqual(firstVector[1].jVal, 2);
            assert.deepEqual(firstVector[1].kVal, 1);
            assert.deepEqual(firstVector[1].layer, 1);
            // examine second vector
            var secondVector = firstLayer[1];
            assert.deepEqual(secondVector[0].jVal, 1);
            assert.deepEqual(secondVector[0].kVal, 2);
            assert.deepEqual(secondVector[0].layer, 1);
            assert.deepEqual(secondVector[1].jVal, 2);
            assert.deepEqual(secondVector[1].kVal, 2);
            assert.deepEqual(secondVector[1].layer, 1);
            // examine final vector
            var thirdVector = firstLayer[2];
            assert.deepEqual(thirdVector[0].jVal, 1);
            assert.deepEqual(thirdVector[0].kVal, 3);
            assert.deepEqual(thirdVector[0].layer, 1);
            assert.deepEqual(thirdVector[1].jVal, 2);
            assert.deepEqual(thirdVector[1].kVal, 3);
            assert.deepEqual(thirdVector[1].layer, 1);
        });
        it("going into second layer", function () {
            var secondLayer = net._weights[1];
            assert.deepEqual(secondLayer.length, 3);
        });
        it("examining weights of second layer", function () {
            var secondLayer = net._weights[1];
            // first ensure that there are the correct number of _weights in each vector of the first layer
            for (var _i = 0, secondLayer_2 = secondLayer; _i < secondLayer_2.length; _i++) {
                var vector = secondLayer_2[_i];
                assert.deepEqual(vector.length, 2);
            }
            // examine first vector
            var firstVector = secondLayer[0]; // biased node
            assert.deepEqual(firstVector[0].jVal, 1);
            assert.deepEqual(firstVector[0].kVal, 1);
            assert.deepEqual(firstVector[0].layer, 2);
            assert.deepEqual(firstVector[1].jVal, 2);
            assert.deepEqual(firstVector[1].kVal, 1);
            assert.deepEqual(firstVector[1].layer, 2);
            // examine second vector
            var secondVector = secondLayer[1];
            assert.deepEqual(secondVector[0].jVal, 1);
            assert.deepEqual(secondVector[0].kVal, 2);
            assert.deepEqual(secondVector[0].layer, 2);
            assert.deepEqual(secondVector[1].jVal, 2);
            assert.deepEqual(secondVector[1].kVal, 2);
            assert.deepEqual(secondVector[1].layer, 2);
            // examine final vector
            var thirdVector = secondLayer[2];
            assert.deepEqual(thirdVector[0].jVal, 1);
            assert.deepEqual(thirdVector[0].kVal, 3);
            assert.deepEqual(thirdVector[0].layer, 2);
            assert.deepEqual(thirdVector[1].jVal, 2);
            assert.deepEqual(thirdVector[1].kVal, 3);
            assert.deepEqual(thirdVector[1].layer, 2);
        });
        it("going into final layer", function () {
            var finalLayer = net._weights[2];
            assert.deepEqual(finalLayer.length, 3);
        });
        it("examining _weights of final layer", function () {
            var finalLayer = net._weights[2];
            // first ensure that there are the correct number of _weights in each vector of the first layer
            for (var _i = 0, finalLayer_3 = finalLayer; _i < finalLayer_3.length; _i++) {
                var vector = finalLayer_3[_i];
                assert.deepEqual(vector.length, 1);
            }
            // examine first vector
            var firstVector = finalLayer[0]; // biased node
            assert.deepEqual(firstVector[0].jVal, 1);
            assert.deepEqual(firstVector[0].kVal, 1);
            assert.deepEqual(firstVector[0].layer, 3);
            // examine second vector
            var secondVector = finalLayer[1];
            assert.deepEqual(secondVector[0].jVal, 1);
            assert.deepEqual(secondVector[0].kVal, 2);
            assert.deepEqual(secondVector[0].layer, 3);
            // examine final vector
            var thirdVector = finalLayer[2];
            assert.deepEqual(thirdVector[0].jVal, 1);
            assert.deepEqual(thirdVector[0].kVal, 3);
            assert.deepEqual(thirdVector[0].layer, 3);
        });
    });
});
