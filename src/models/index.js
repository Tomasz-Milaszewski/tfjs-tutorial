import * as tf from '@tensorflow/tfjs';

//Create sequential model
const model = tf.sequential({
  layers: [
    tf.layers.dense({inputShape: [784], units: 32, activation: 'relu'}),
    tf.layers.dense({units: 10, activation: 'softmax'}),
  ]
 });
 model.summary();
 // No of params for each layer is:
 // layer one: 784(inputShape) * 32(units) + 32 = 25120
 // layer two: 32(units of previous layer) * 10(units) + 10 = 330