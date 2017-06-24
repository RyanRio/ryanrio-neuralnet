import { Example } from './example';

describe("TEST", () => {
    it("test", () => {
        const example = new Example();
        expect(example.a).toBe(2);
    });
});
