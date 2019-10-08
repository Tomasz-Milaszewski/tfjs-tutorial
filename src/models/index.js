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

 //Create a sequential model via add() method
const model2 = tf.sequential();
model2.add(tf.layers.dense({inputShape: [784], units: 32, activation: 'relu'})); 
model2.add(tf.layers.dense({units: 10, activation: 'softmax'}));

// Create the same model with an arbitrary graph of layers, by connecting them via the apply() method.
const input = tf.input({shape: [784]});
const dense1 = tf.layers.dense({units: 32, activation: 'relu'}).apply(input);
const dense2 = tf.layers.dense({units: 10, activation: 'softmax'}).apply(dense1);
const model3 = tf.model({inputs: input, outputs: dense2});
model3.summary();

//Use apply() to output a concrete tensor
const t = tf.tensor([-2, 1, 0, 5]);
const o = tf.layers.activation({activation: 'relu'}).apply(t);
o.print(); // [0, 1, 0, 5]