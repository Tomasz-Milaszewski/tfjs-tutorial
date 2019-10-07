import * as tf from '@tensorflow/tfjs';
// Create a rank-2 tensor (matrix) matrix tensor from a multidimensional array.
const a = tf.tensor([[1, 2], [3, 4, 5, 6], [7, 8]]);
console.log('rank:', a.rank);
console.log('shape:', a.shape);
a.print();

// Or you can create a tensor from a flat array and specify a shape.
const shape = [4, 1];
const b = tf.tensor([1, 2, 3, 4], shape);
console.log('rank:', b.rank);
console.log('shape:', b.shape);
b.print();

const c = tf.tensor([[1, 2, 3], [3, 4, 5]], [2, 3], 'int32');
console.log('rank:', c.rank);
console.log('shape:', c.shape);
console.log('dtype', c.dtype);
c.print();