export interface LoggerInterface {
    log(logMessage: string): void;
}
import { NeuralNet } from './neural-net';
export declare abstract class Logger implements LoggerInterface {
    constructor();
    log(logMessage: string): void;
}
export declare class ManagerLogger implements Logger {
    net: NeuralNet;
    timesSaved: number;
    constructor(net: NeuralNet);
    log(logMessage: string): void;
}
