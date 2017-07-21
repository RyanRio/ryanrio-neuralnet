import { NeuralNet } from './neural-net';
import * as assert from 'assert';



describe("neural-net(rw)", () => {
    describe("weight matrix assembly test", () => {
        let net = new NeuralNet(2, 1, 2, 1);
        it("should get created correctly", () => {
            assert.deepEqual(net.weightMatrices.length, 2);
        });
        describe("first matrix should be correct", () => {
            const firstMatrix = net.weightMatrices[0].vectors;
            assert.deepEqual(firstMatrix.length, 3);
            for (let vector of firstMatrix) {
                assert.deepEqual(vector.length, 2);
            }
        });
    });
    describe("weight-getting testing", () => {
        it("should get the correct weight, #1", () => {
            let net = new NeuralNet(2, 1, 2, 1);
            let weight = net.weightVal(1, 3, 2);
            assert.deepEqual(weight.jVal, 1);
            assert.deepEqual(weight.kVal, 3);
            assert.deepEqual(weight.layer, 1);
        });

        it("should get the correct weight, #2", () => {
            let net = new NeuralNet(2, 1, 2, 1);
            let weight = net.weightVal(1, 3, 3);
            assert.deepEqual(weight.jVal, 2);
            assert.deepEqual(weight.kVal, 3);
            assert.deepEqual(weight.layer, 1);
        });
    });
    describe("_weights-test with one hidden layer", () => {
        let net = new NeuralNet(2, 1, 2, 1);
        it("number of weight layers", () => {
            assert.deepEqual(net._weights.length, 2);
        });
        it("going into first layer", () => {
            const firstLayer = net._weights[0];
            assert.deepEqual(firstLayer.length, 3);
        });
        it("examining _weights of first layer", () => {
            const firstLayer = net._weights[0];
            // first ensure that there are the correct number of _weights in each vector of the first layer
            for (let vector of firstLayer) {
                assert.deepEqual(vector.length, 2);
            }
            // examine first vector
            const firstVector = firstLayer[0]; // biased node
            assert.deepEqual(firstVector[0].jVal, 1);
            assert.deepEqual(firstVector[0].kVal, 1);
            assert.deepEqual(firstVector[0].layer, 1);

            assert.deepEqual(firstVector[1].jVal, 2);
            assert.deepEqual(firstVector[1].kVal, 1);
            assert.deepEqual(firstVector[1].layer, 1);

            // examine second vector
            const secondVector = firstLayer[1];
            assert.deepEqual(secondVector[0].jVal, 1);
            assert.deepEqual(secondVector[0].kVal, 2);
            assert.deepEqual(secondVector[0].layer, 1);

            assert.deepEqual(secondVector[1].jVal, 2);
            assert.deepEqual(secondVector[1].kVal, 2);
            assert.deepEqual(secondVector[1].layer, 1);

            // examine final vector
            const thirdVector = firstLayer[2];
            assert.deepEqual(thirdVector[0].jVal, 1);
            assert.deepEqual(thirdVector[0].kVal, 3);
            assert.deepEqual(thirdVector[0].layer, 1);

            assert.deepEqual(thirdVector[1].jVal, 2);
            assert.deepEqual(thirdVector[1].kVal, 3);
            assert.deepEqual(thirdVector[1].layer, 1);
        });

        it("going into final layer", () => {
            const finalLayer = net._weights[1];
            assert.deepEqual(finalLayer.length, 3);
        });
        it("examining _weights of final layer", () => {
            const finalLayer = net._weights[1];
            // first ensure that there are the correct number of _weights in each vector of the first layer
            for (let vector of finalLayer) {
                assert.deepEqual(vector.length, 2);
            }
            // examine first vector
            const firstVector = finalLayer[0]; // biased node
            assert.deepEqual(firstVector[0].jVal, 1);
            assert.deepEqual(firstVector[0].kVal, 1);
            assert.deepEqual(firstVector[0].layer, 2);

            assert.deepEqual(firstVector[1].jVal, 2);
            assert.deepEqual(firstVector[1].kVal, 1);
            assert.deepEqual(firstVector[1].layer, 2);

            // examine second vector
            const secondVector = finalLayer[1];
            assert.deepEqual(secondVector[0].jVal, 1);
            assert.deepEqual(secondVector[0].kVal, 2);
            assert.deepEqual(secondVector[0].layer, 2);

            assert.deepEqual(secondVector[1].jVal, 2);
            assert.deepEqual(secondVector[1].kVal, 2);
            assert.deepEqual(secondVector[1].layer, 2);

            // examine final vector
            const thirdVector = finalLayer[2];
            assert.deepEqual(thirdVector[0].jVal, 1);
            assert.deepEqual(thirdVector[0].kVal, 3);
            assert.deepEqual(thirdVector[0].layer, 2);

            assert.deepEqual(thirdVector[1].jVal, 2);
            assert.deepEqual(thirdVector[1].kVal, 3);
            assert.deepEqual(thirdVector[1].layer, 2);
        });

    });



    describe("_weights-test with two layers", () => {
        let net = new NeuralNet(2, 2, 2, 1);
        it("number of weight layers", () => {
            assert.deepEqual(net._weights.length, 3);
        });
        it("going into first layer", () => {
            const firstLayer = net._weights[0];
            assert.deepEqual(firstLayer.length, 3);
        });
        it("examining _weights of first layer", () => {
            const firstLayer = net._weights[0];
            // first ensure that there are the correct number of _weights in each vector of the first layer
            for (let vector of firstLayer) {
                assert.deepEqual(vector.length, 2);
            }
            // examine first vector
            const firstVector = firstLayer[0]; // biased node
            assert.deepEqual(firstVector[0].jVal, 1);
            assert.deepEqual(firstVector[0].kVal, 1);
            assert.deepEqual(firstVector[0].layer, 1);

            assert.deepEqual(firstVector[1].jVal, 2);
            assert.deepEqual(firstVector[1].kVal, 1);
            assert.deepEqual(firstVector[1].layer, 1);

            // examine second vector
            const secondVector = firstLayer[1];
            assert.deepEqual(secondVector[0].jVal, 1);
            assert.deepEqual(secondVector[0].kVal, 2);
            assert.deepEqual(secondVector[0].layer, 1);

            assert.deepEqual(secondVector[1].jVal, 2);
            assert.deepEqual(secondVector[1].kVal, 2);
            assert.deepEqual(secondVector[1].layer, 1);

            // examine final vector
            const thirdVector = firstLayer[2];
            assert.deepEqual(thirdVector[0].jVal, 1);
            assert.deepEqual(thirdVector[0].kVal, 3);
            assert.deepEqual(thirdVector[0].layer, 1);

            assert.deepEqual(thirdVector[1].jVal, 2);
            assert.deepEqual(thirdVector[1].kVal, 3);
            assert.deepEqual(thirdVector[1].layer, 1);
        });
        it("going into second layer", () => {
            const secondLayer = net._weights[1];
            assert.deepEqual(secondLayer.length, 3);
        });
        it("examining _weights of second layer", () => {
            const secondLayer = net._weights[1];
            // first ensure that there are the correct number of _weights in each vector of the first layer
            for (let vector of secondLayer) {
                assert.deepEqual(vector.length, 2);
            }
            // examine first vector
            const firstVector = secondLayer[0]; // biased node
            assert.deepEqual(firstVector[0].jVal, 1);
            assert.deepEqual(firstVector[0].kVal, 1);
            assert.deepEqual(firstVector[0].layer, 2);

            assert.deepEqual(firstVector[1].jVal, 2);
            assert.deepEqual(firstVector[1].kVal, 1);
            assert.deepEqual(firstVector[1].layer, 2);

            // examine second vector
            const secondVector = secondLayer[1];
            assert.deepEqual(secondVector[0].jVal, 1);
            assert.deepEqual(secondVector[0].kVal, 2);
            assert.deepEqual(secondVector[0].layer, 2);

            assert.deepEqual(secondVector[1].jVal, 2);
            assert.deepEqual(secondVector[1].kVal, 2);
            assert.deepEqual(secondVector[1].layer, 2);

            // examine final vector
            const thirdVector = secondLayer[2];
            assert.deepEqual(thirdVector[0].jVal, 1);
            assert.deepEqual(thirdVector[0].kVal, 3);
            assert.deepEqual(thirdVector[0].layer, 2);

            assert.deepEqual(thirdVector[1].jVal, 2);
            assert.deepEqual(thirdVector[1].kVal, 3);
            assert.deepEqual(thirdVector[1].layer, 2);
        });

        it("going into final layer", () => {
            const finalLayer = net._weights[2];
            assert.deepEqual(finalLayer.length, 3);
        });
        it("examining _weights of final layer", () => {
            const finalLayer = net._weights[2];
            // first ensure that there are the correct number of _weights in each vector of the first layer
            for (let vector of finalLayer) {
                assert.deepEqual(vector.length, 2);
            }
            // examine first vector
            const firstVector = finalLayer[0]; // biased node
            assert.deepEqual(firstVector[0].jVal, 1);
            assert.deepEqual(firstVector[0].kVal, 1);
            assert.deepEqual(firstVector[0].layer, 3);

            assert.deepEqual(firstVector[1].jVal, 2);
            assert.deepEqual(firstVector[1].kVal, 1);
            assert.deepEqual(firstVector[1].layer, 3);

            // examine second vector
            const secondVector = finalLayer[1];
            assert.deepEqual(secondVector[0].jVal, 1);
            assert.deepEqual(secondVector[0].kVal, 2);
            assert.deepEqual(secondVector[0].layer, 3);

            assert.deepEqual(secondVector[1].jVal, 2);
            assert.deepEqual(secondVector[1].kVal, 2);
            assert.deepEqual(secondVector[1].layer, 3);

            // examine final vector
            const thirdVector = finalLayer[2];
            assert.deepEqual(thirdVector[0].jVal, 1);
            assert.deepEqual(thirdVector[0].kVal, 3);
            assert.deepEqual(thirdVector[0].layer, 3);

            assert.deepEqual(thirdVector[1].jVal, 2);
            assert.deepEqual(thirdVector[1].kVal, 3);
            assert.deepEqual(thirdVector[1].layer, 3);
        });
    });

    describe("_weights test with non-equal input elements and output elements", () => {
        let net = new NeuralNet(2, 2, 1, 1);
        it("number of weight layers", () => {
            assert.deepEqual(net._weights.length, 3);
        });
        it("going into first layer", () => {
            const firstLayer = net._weights[0];
            assert.deepEqual(firstLayer.length, 3);
        });
        it("examining _weights of first layer", () => {
            const firstLayer = net._weights[0];
            // first ensure that there are the correct number of _weights in each vector of the first layer
            for (let vector of firstLayer) {
                assert.deepEqual(vector.length, 2);
            }
            // examine first vector
            const firstVector = firstLayer[0]; // biased node
            assert.deepEqual(firstVector[0].jVal, 1);
            assert.deepEqual(firstVector[0].kVal, 1);
            assert.deepEqual(firstVector[0].layer, 1);

            assert.deepEqual(firstVector[1].jVal, 2);
            assert.deepEqual(firstVector[1].kVal, 1);
            assert.deepEqual(firstVector[1].layer, 1);

            // examine second vector
            const secondVector = firstLayer[1];
            assert.deepEqual(secondVector[0].jVal, 1);
            assert.deepEqual(secondVector[0].kVal, 2);
            assert.deepEqual(secondVector[0].layer, 1);

            assert.deepEqual(secondVector[1].jVal, 2);
            assert.deepEqual(secondVector[1].kVal, 2);
            assert.deepEqual(secondVector[1].layer, 1);

            // examine final vector
            const thirdVector = firstLayer[2];
            assert.deepEqual(thirdVector[0].jVal, 1);
            assert.deepEqual(thirdVector[0].kVal, 3);
            assert.deepEqual(thirdVector[0].layer, 1);

            assert.deepEqual(thirdVector[1].jVal, 2);
            assert.deepEqual(thirdVector[1].kVal, 3);
            assert.deepEqual(thirdVector[1].layer, 1);
        });
        it("going into second layer", () => {
            const secondLayer = net._weights[1];
            assert.deepEqual(secondLayer.length, 3);
        });
        it("examining _weights of second layer", () => {
            const secondLayer = net._weights[1];
            // first ensure that there are the correct number of _weights in each vector of the first layer
            for (let vector of secondLayer) {
                assert.deepEqual(vector.length, 2);
            }
            // examine first vector
            const firstVector = secondLayer[0]; // biased node
            assert.deepEqual(firstVector[0].jVal, 1);
            assert.deepEqual(firstVector[0].kVal, 1);
            assert.deepEqual(firstVector[0].layer, 2);

            assert.deepEqual(firstVector[1].jVal, 2);
            assert.deepEqual(firstVector[1].kVal, 1);
            assert.deepEqual(firstVector[1].layer, 2);

            // examine second vector
            const secondVector = secondLayer[1];
            assert.deepEqual(secondVector[0].jVal, 1);
            assert.deepEqual(secondVector[0].kVal, 2);
            assert.deepEqual(secondVector[0].layer, 2);

            assert.deepEqual(secondVector[1].jVal, 2);
            assert.deepEqual(secondVector[1].kVal, 2);
            assert.deepEqual(secondVector[1].layer, 2);

            // examine final vector
            const thirdVector = secondLayer[2];
            assert.deepEqual(thirdVector[0].jVal, 1);
            assert.deepEqual(thirdVector[0].kVal, 3);
            assert.deepEqual(thirdVector[0].layer, 2);

            assert.deepEqual(thirdVector[1].jVal, 2);
            assert.deepEqual(thirdVector[1].kVal, 3);
            assert.deepEqual(thirdVector[1].layer, 2);
        });

        it("going into final layer", () => {
            const finalLayer = net._weights[2];
            assert.deepEqual(finalLayer.length, 3);
        });
        it("examining _weights of final layer", () => {
            const finalLayer = net._weights[2];
            // first ensure that there are the correct number of _weights in each vector of the first layer
            for (let vector of finalLayer) {
                assert.deepEqual(vector.length, 1);
            }
            // examine first vector
            const firstVector = finalLayer[0]; // biased node
            assert.deepEqual(firstVector[0].jVal, 1);
            assert.deepEqual(firstVector[0].kVal, 1);
            assert.deepEqual(firstVector[0].layer, 3);

            // examine second vector
            const secondVector = finalLayer[1];
            assert.deepEqual(secondVector[0].jVal, 1);
            assert.deepEqual(secondVector[0].kVal, 2);
            assert.deepEqual(secondVector[0].layer, 3);

            // examine final vector
            const thirdVector = finalLayer[2];
            assert.deepEqual(thirdVector[0].jVal, 1);
            assert.deepEqual(thirdVector[0].kVal, 3);
            assert.deepEqual(thirdVector[0].layer, 3);
        });
    });
});
