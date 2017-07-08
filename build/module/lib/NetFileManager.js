import * as fs from 'fs';
import { ManagerLogger } from './Logger';
var NetFileManager = (function () {
    /**
     * Upon contstruction the net is saved
     * @param net;
     */
    function NetFileManager(net) {
        this.net = net;
        this.logger = new ManagerLogger(this.net);
        this.save();
    }
    NetFileManager.prototype.save = function () {
        var _this = this;
        this.logger.log("Net was just saved: " + this.logger.timesSaved);
        fs.exists("NetJSON/net.json", (function (exists) {
            if (exists) {
                fs.unlink("NetJSON/net.json", function (err) {
                    if (err) {
                        throw err;
                    }
                    else {
                        console.log("File exists, deleting and rewriting");
                        fs.writeFile("NetJSON/net.json", _this.net.JSON, function (err) {
                            if (err) {
                                throw err;
                            }
                        });
                    }
                });
            }
            else {
                console.log("File doesn't exist, writing");
                fs.writeFile("NetJSON/net.json", _this.net.JSON, function (err) {
                    if (err) {
                        throw err;
                    }
                });
            }
        }));
    };
    return NetFileManager;
}());
export { NetFileManager };
