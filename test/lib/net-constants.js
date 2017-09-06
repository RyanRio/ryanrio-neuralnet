"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.activateFuncDefault = function (input) {
    var activatedInput = (1 / (1 + Math.exp(-1 * input)));
    return activatedInput;
};
