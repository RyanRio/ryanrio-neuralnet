import * as fs from 'fs';

export interface LoggerInterface {
    log(logMessage: string): void;
}

import { NeuralNet } from './neural-net';

abstract class Logger implements LoggerInterface {
    constructor() {
        const thisa = this;
        setInterval(function () {
            thisa.log("");
        }, 3000);
    }

    public log(logMessage: string) {
        fs.writeFile('log.txt', `LOGGING\n`, err => {
            if (err) {
                throw err;
            }
        });
    }
}

export class ManagerLogger implements Logger {
    net: NeuralNet;
    timesSaved: number;
    constructor(net: NeuralNet) {
        this.timesSaved = 1;
        this.net = net;
    }

    public log(logMessage: string) {
        this.timesSaved++;
        fs.appendFile('logs/log.txt', logMessage + "\n", err => {
            if (err) {
                throw err;
            }
        });
    }
}
