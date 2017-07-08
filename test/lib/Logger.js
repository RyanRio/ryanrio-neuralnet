"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var fs = require("fs");
var Logger = (function () {
    function Logger() {
        var thisa = this;
        setInterval(function () {
            thisa.log("");
        }, 3000);
    }
    Logger.prototype.log = function (logMessage) {
        fs.writeFile('log.txt', "LOGGING\n", function (err) {
            if (err) {
                throw err;
            }
        });
    };
    return Logger;
}());
var ManagerLogger = (function () {
    function ManagerLogger(net) {
        this.timesSaved = 1;
        this.net = net;
    }
    ManagerLogger.prototype.log = function (logMessage) {
        this.timesSaved++;
        fs.appendFile('logs/log.txt', logMessage + "\n", function (err) {
            if (err) {
                throw err;
            }
        });
    };
    return ManagerLogger;
}());
exports.ManagerLogger = ManagerLogger;
