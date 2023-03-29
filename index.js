import { DATA } from '../dataset';
import { getAccuracy } from './getAccuracy';

const SPLIT = 99;
const trainData = DATA.slice(0, SPLIT);
const testData = DATA.slice(SPLIT + 1);

const net = new brain.NeuralNetwork({
  activation: 'sigmoid', // activation function
  hiddenLayers: [2],
  iterations: 20000,
  learningRate: 0.5 
});
net.train(trainData);

const accuracy = getAccuracy(net, testData);
console.log('accuracy: ', accuracy);