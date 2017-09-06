export const activateFuncDefault = (input: number): number => {
    const activatedInput = (1 / (1 + Math.exp(-1 * input)));
    return activatedInput;
};
