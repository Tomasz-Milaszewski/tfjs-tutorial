import * as tf from '@tensorflow/tfjs';
// Create a rank-2 tensor (matrix) matrix tensor from a multidimensional array.
const a = tf.tensor([[1, 2], [3, 4, 5, 6], [7, 8]]);
console.log('rank:', a.rank);
console.log('shape:', a.shape);
a.print();

// Or you can create a tensor from a flat array and specify a shape.
const shape = [2, 2];
const b = tf.tensor([1, 2, 3, 4], shape);
console.log('rank:', b.rank);
console.log('shape:', b.shape);
b.print();

const c = tf.tensor([[1, 2, 3], [3, 4, 5]], [2, 3], 'int32');
console.log('rank:', c.rank);
console.log('shape:', c.shape);
console.log('dtype', c.dtype);
c.print();

const d = b.reshape([1,4]);
console.log('shape:', d.shape);
d.print();

// Returns the multi dimensional array of values.
b.array().then(array => console.log(array));
// Returns the flattened data that backs the tensor.
b.data().then(data => console.log(data));

const x = tf.tensor([1, 2, 3, 4]);
const y = x.square();  // equivalent to tf.square(x)
y.print();

const e = tf.tensor([1, 2, 3, 4]);
const f = tf.tensor([10, 20, 30, 40]);
const z = e.add(f);  // equivalent to tf.add(e, f)
z.print();

const aa = tf.tensor([[1, 2], [3, 4]]);
aa.dispose(); // Equivalent to tf.dispose(aa)

const aaa = tf.tensor([[1, 2], [3, 4]]);
const yy = tf.tidy(() => {
  const result = aaa.square().log().neg(); // log and square will automatically be disposed
  return result;
});

console.log(tf.memory());