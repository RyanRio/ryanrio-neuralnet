import * as fs from 'fs';
import { ManagerLogger } from './Logger';
import { NeuralNet } from './neural-net';

export class NetFileManager {
    logger: ManagerLogger;
    net: NeuralNet;
    /**
     * Upon contstruction the net is saved
     * @param net;
     */
    constructor(net: NeuralNet) {
        this.net = net;
        this.logger = new ManagerLogger(this.net);
        this.save();
    }

    public save() {
        this.logger.log(`Net was just saved: ${this.logger.timesSaved}`);
        fs.exists("NetJSON/net.json", (exists => {
            if (exists) {
                fs.unlink("NetJSON/net.json", err => {
                    if (err) {
                        throw err;
                    }
                    else {
                        console.log("File exists, deleting and rewriting");
                        fs.writeFile("NetJSON/net.json", this.net.JSON, err => {
                            if (err) {
                                throw err;
                            }
                        });
                    }
                });
            }
            else {
                console.log("File doesn't exist, writing");
                fs.writeFile("NetJSON/net.json", this.net.JSON, err => {
                    if (err) {
                        throw err;
                    }
                });
            }
        }));
    }
}
