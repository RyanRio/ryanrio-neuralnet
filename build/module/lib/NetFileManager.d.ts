import { ManagerLogger } from './Logger';
import { NeuralNet } from './neural-net';
export declare class NetFileManager {
    logger: ManagerLogger;
    net: NeuralNet;
    /**
     * Upon contstruction the net is saved
     * @param net;
     */
    constructor(net: NeuralNet);
    save(): void;
}
