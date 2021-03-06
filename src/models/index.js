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

//Log model weights
model.weights.forEach(w => {
  console.log(w.name, w.shape);
});
// Output is:
//> dense_Dense1/kernel [784, 32]
//> dense_Dense1/bias [32]
//> dense_Dense2/kernel [32, 10]
//> dense_Dense2/bias [10]
// There are 4 weights in total, 2 per dense layer.
// This is expected since dense layers represent a function that maps the input tensor x to an output tensor y via the equation y = Ax + b where A (the kernel) and b (the bias) are parameters of the dense layer.

// Demonstration only - overriding weights auto-initialized by layers API 
// model.weights.forEach(w => {
//   const newVals = tf.randomNormal(w.shape);
//   // w.val is an instance of tf.Variable
//   w.val.assign(newVals);
// });

//Compile model
model.compile({
  optimizer: 'sgd',
  loss: 'categoricalCrossentropy',
  metrics: ['accuracy']
});

//Training model with model.fit()
// Generate dummy data.
const data = tf.randomNormal([100, 784]);
const labels = tf.randomUniform([100, 10]);

function onBatchEnd(batch, logs) {
  console.log('Accuracy', logs.acc);
}

// Train for 5 epochs with batch size of 32.
// model.fit(data, labels, {
//    epochs: 5,
//    batchSize: 32,
//    callbacks: {onBatchEnd}
//  }).then(info => {
//    console.log('Final accuracy', info.history.acc);
//  });

//Train the model with fitDataset()
function* data2() {
  for (let i = 0; i < 100; i++) {
    // Generate one sample at a time.
    yield tf.randomNormal([784]);
  }
 }
 
 function* labels2() {
  for (let i = 0; i < 100; i++) {
    // Generate one sample at a time.
    yield tf.randomUniform([10]);
  }
 }
 
 const xs = tf.data.generator(data2);
 const ys = tf.data.generator(labels2);
 // We zip the data and labels together, shuffle and batch 32 samples at a time.
 const ds = tf.data.zip({xs, ys}).shuffle(100 /* bufferSize */).batch(32);
 
 // Train the model for 5 epochs.
 model.fitDataset(ds, {epochs: 5}).then(info => {
  console.log('Accuracy', info.history.acc);
 });

// Predict 3 random samples.
const prediction = model.predict(tf.randomNormal([3, 784]));
prediction.print();

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

//Save and load to local storage
//const saveResult = await model.save('localstorage://my-model-1');
//const model = await tf.loadLayersModel('localstorage://my-model-1');

//Defining custom layer
class SquaredSumLayer extends tf.layers.Layer {
  constructor() {
    super({});
  }
  // In this case, the output is a scalar.
  computeOutputShape(inputShape) { return []; }
 
  // call() is where we do the computation.
  call(input, kwargs) { return input.square().sum();}
 
  // Every layer needs a unique name.
  getClassName() { return 'SquaredSum'; }
 }

const tt = tf.tensor([-2, 1, 0, 5]);
const oo = new SquaredSumLayer().apply(tt);
oo.print();